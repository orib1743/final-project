import os
import pandas as pd
import re
from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from wordcloud import WordCloud
from skopt import gp_minimize
from skopt.space import Integer
import numpy as np
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


#  שלב 1: טעינת מודל מאומן מראש
pretrained_model_path = r'C:\Users\yifat\לימודים\שנה ג\סמסטר א\זיהוי דיבור עיבוד שפה\NLP_Transfer_Learning_Method\GoogleNews-vectors-negative300.bin'
pretrained_model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)

#  שלב 2: הגדרת נתיבים ===
file_path_2017 = r'C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017_fixed.xlsx'
file_path_2018 = r'C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018_fixed.xlsx'
output_folder = r'C:\Users\yifat\PycharmProjects\Final-Project\NLP_Transfer_Learning_Methods\Word2Vec\Visualizations_FineTune'
combined_output_path = r'C:\Users\yifat\PycharmProjects\Final-Project\NLP_Transfer_Learning_Methods\Word2Vec\Combined_Output_FineTune.xlsx'

#  שלב 3: הגדרת Stopwords ===
stop_words = set(stopwords.words('english'))
additional_stop_words = {'title', 'section', 'pub', 'repealed', 'part,', 'chapter', 'subpart', 'subchapter', 'subtitle',
                         'div'}
stop_words.update(additional_stop_words)


# === שלב 4: ניקוי וטיוב טקסט ===
def clean_content(text):
    text = re.sub(r'^(subtitle|chapter|subchapter|part|subpart)\s?[a-zA-Z\d]*\s?[–—]', '', text,
                  flags=re.IGNORECASE).strip()
    text = re.sub(r'[–—]', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# === שלב 5: עיבוד נתונים ===
def preprocess_and_tokenize(df, content_column):
    df['cleaned_content'] = df[content_column].apply(lambda x: clean_content(str(x)))
    df['tokens'] = df['cleaned_content'].apply(
        lambda x: [word for word in x.lower().split() if
                   word not in stop_words and len(word) > 1])
    return df


# === שלב 6: Fine-Tuning למודל Word2Vec ===
def fine_tune_word2vec(tokens, vector_size=300, window=5, min_count=2, sg=1, epochs=5):
    print("[INFO] Fine-Tuning Word2Vec on legal corpus...")

    # שלב 6.1: יצירת המודל
    print("[DEBUG] Creating Word2Vec model instance...")
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        hs=0,
        negative=5,
        workers=4,
        epochs=epochs
    )

    # שלב 6.2: עדכון אוצר המילים
    print("[DEBUG] Building vocabulary...")
    model.build_vocab(tokens, update=False)

    # שלב 6.3: הוספת וקטורים מהמודל המאומן מראש
    print("[DEBUG] Adding pretrained vectors...")
    for word in pretrained_model.key_to_index.keys():
        if word in model.wv:
            model.wv[word] = pretrained_model[word]

    # שלב 6.4: אימון המודל
    print("[DEBUG] Training model on legal corpus...")
    model.train(tokens, total_examples=len(tokens), epochs=epochs)

    print("[INFO] Fine-Tuning Completed.")
    return model


# === שלב 7: Bayesian Optimization עבור n_clusters ===
def objective(n_clusters, word_vectors):
    n_clusters = int(n_clusters[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(word_vectors)
    labels = kmeans.labels_

    if len(set(labels)) > 1:
        silhouette = silhouette_score(word_vectors, labels)
        return -silhouette
    else:
        return -1


def optimize_n_clusters(word_vectors):
    search_space = [Integer(2, 10, name='n_clusters')]
    result = gp_minimize(
        func=lambda n_clusters: objective(n_clusters, word_vectors),
        dimensions=search_space,
        n_calls=10,
        random_state=42
    )
    return result.x[0]


# === שלב 8: Clustering ו-Word Cloud ===
def cluster_and_visualize(tokens, model, year):
    # ספירת המילים ומציאת 100 המילים הנפוצות ביותר
    word_counts = Counter([word for token_list in tokens for word in token_list])
    print(f"[INFO] Number of words before filtering: {len(word_counts)}")
    top_words = [word for word, count in word_counts.most_common(100)]
    word_vectors = [model.wv[word] for word in top_words if word in model.wv]

    words_not_in_model = [word for word in top_words if word not in model.wv]
    print(f"[INFO] Words not in pretrained model: {len(words_not_in_model)}")

    optimal_clusters = optimize_n_clusters(word_vectors)
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42).fit(word_vectors)
    labels = kmeans.labels_

    clusters = {i: [] for i in range(optimal_clusters)}
    for i, word in enumerate(top_words):
        if word in model.wv:
            clusters[labels[i]].append(word)

    os.makedirs(output_folder, exist_ok=True)
    for cluster_id, words in clusters.items():
        if words:  # בדיקה אם יש מילים בקבוצה
            print(f"[INFO] Creating Word Cloud for Year {year}, Cluster {cluster_id} with {len(words)} words.")
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(' '.join(words))
            file_name = os.path.join(output_folder, f"Year_{year}Cluster{cluster_id}.png")
            wordcloud.to_file(file_name)
            print(f"[INFO] Word Cloud saved: {file_name}")
        else:
            print(f"[WARNING] No words found for Year {year}, Cluster {cluster_id}. Skipping Word Cloud.")

    combined_data = []
    for word in top_words:
        combined_data.append({'Year': year, 'Word': word, 'Frequency': word_counts[word]})

    return pd.DataFrame(combined_data)


# === שלב 9: קריאות לפונקציות ===
data_2017 = pd.read_excel(file_path_2017)
data_2018 = pd.read_excel(file_path_2018)

data_2017 = preprocess_and_tokenize(data_2017, 'content')
data_2018 = preprocess_and_tokenize(data_2018, 'content')

model_2017 = fine_tune_word2vec(data_2017['tokens'])
model_2018 = fine_tune_word2vec(data_2018['tokens'])

df_2017 = cluster_and_visualize(data_2017['tokens'], model_2017, 2017)
df_2018 = cluster_and_visualize(data_2018['tokens'], model_2018, 2018)

combined_df = pd.concat([df_2017, df_2018], ignore_index=True)
combined_df.to_excel(combined_output_path, index=False)
print(f"[INFO] Combined Output saved at: {combined_output_path}")