import os
import pandas as pd
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from wordcloud import WordCloud
from skopt import gp_minimize
from skopt.space import Integer
import numpy as np
from collections import Counter

# הגדרת נתיבים
file_path_2017 = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017_fixed.xlsx'
file_path_2018 = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018_fixed.xlsx'
output_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\NLP_Transfer_Learning_Methods\Word2Vec\Combined_Output.xlsx'
output_folder = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\NLP_Transfer_Learning_Methods\Word2Vec\Visualizations'

# הגדרת stopwords
stop_words = set(stopwords.words('english'))
additional_stop_words = {'title', 'section', 'pub', 'repealed', 'part,', 'chapter', 'subpart', 'subchapter', 'subtitle',
                         'div'}
stop_words.update(additional_stop_words)


# ניקוי וטיוב טקסט
def clean_content(text):
    text = re.sub(r'^(subtitle|chapter|subchapter|part|subpart)\s?[a-zA-Z\d]*\s?[–—]', '', text,
                  flags=re.IGNORECASE).strip()
    text = re.sub(r'[–—]', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# עיבוד נתונים
def preprocess_and_tokenize(df, content_column):
    print(f"[INFO] Preprocessing data for column '{content_column}'...", flush=True)
    df['cleaned_content'] = df[content_column].apply(lambda x: clean_content(str(x)))
    df['tokens'] = df['cleaned_content'].apply(
        lambda x: [word for word in x.lower().split() if word not in stop_words and len(word) > 2])
    print("[INFO] Preprocessing completed.", flush=True)
    return df


# פונקציית מטרה לאופטימיזציה
def objective(params, tokens):
    vector_size, window, min_count, sg = params
    vector_size, window, min_count, sg = int(vector_size), int(window), int(min_count), int(sg)

    sampled_tokens = tokens.sample(frac=0.5, random_state=42).tolist()

    print(f"[INFO] Training Word2Vec with vector_size={vector_size}, window={window}, min_count={min_count}, sg={sg}",
          flush=True)
    model = Word2Vec(
        sentences=sampled_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        hs=0,
        negative=5,
        workers=4,
        epochs=5
    )
    print("[INFO] Word2Vec training completed.", flush=True)

    word_counts = Counter([word for token_list in sampled_tokens for word in token_list])
    top_words = [word for word, count in word_counts.most_common(100)]
    word_vectors = [model.wv[word] for word in top_words if word in model.wv]

    kmeans = KMeans(n_clusters=5, random_state=42).fit(word_vectors)
    labels = kmeans.labels_

    if len(set(labels)) > 1:
        silhouette = silhouette_score(word_vectors, labels)
        return -silhouette
    else:
        return -1


# אופטימיזציה של פרמטרים עם Bayesian Optimization
def optimize_params(tokens, year):
    print(f"[INFO] Starting Bayesian Optimization for {year}...", flush=True)
    search_space = [
        Integer(50, 100, name='vector_size'),
        Integer(2, 10, name='window'),
        Integer(1, 5, name='min_count'),
        Integer(0, 1, name='sg')
    ]
    result = gp_minimize(
        func=lambda params: objective(params, tokens),
        dimensions=search_space,
        n_calls=10,
        random_state=42
    )
    print(f"[INFO] Completed Bayesian Optimization for {year}. Best Params: {result.x}", flush=True)
    return result.x


# Clustering ו-Word Cloud
def cluster_and_visualize(tokens, best_params, year):
    print(f"[INFO] Clustering and Visualizing for {year} with params: {best_params}", flush=True)
    model = Word2Vec(sentences=tokens, vector_size=best_params[0], window=best_params[1],
                     min_count=best_params[2], sg=best_params[3], workers=4, epochs=10)

    word_counts = Counter([word for token_list in tokens for word in token_list])
    top_words = [word for word, count in word_counts.most_common(100)]
    word_vectors = [model.wv[word] for word in top_words if word in model.wv]

    kmeans = KMeans(n_clusters=5, random_state=42).fit(word_vectors)
    labels = kmeans.labels_

    clusters = {i: [] for i in range(5)}
    for i, word in enumerate(top_words):
        if word in model.wv:
            clusters[labels[i]].append(word)

    os.makedirs(output_folder, exist_ok=True)
    for cluster_id, words in clusters.items():
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(' '.join(words))
        file_name = os.path.join(output_folder, f"Year_{year}_Cluster_{cluster_id}.png")
        wordcloud.to_file(file_name)


# === קריאות לפונקציות ===
print("[INFO] Loading data for 2017 and 2018...", flush=True)
data_2017 = pd.read_excel(file_path_2017)
data_2018 = pd.read_excel(file_path_2018)

data_2017 = preprocess_and_tokenize(data_2017, 'content')
data_2018 = preprocess_and_tokenize(data_2018, 'content')

best_params_2017 = optimize_params(data_2017['tokens'], 2017)
best_params_2018 = optimize_params(data_2018['tokens'], 2018)

cluster_and_visualize(data_2017['tokens'], best_params_2017, 2017)
cluster_and_visualize(data_2018['tokens'], best_params_2018, 2018)
