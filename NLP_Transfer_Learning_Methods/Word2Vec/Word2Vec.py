import pandas as pd
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.corpus import stopwords

# Define stopwords
stop_words = set(stopwords.words('english'))

# Add additional custom stopwords
additional_stop_words = {'title', 'section', 'pub', 'repealed', 'part,', 'chapter', 'subpart', 'subchapter', 'pub', 'subtitle', 'div'}
stop_words.update(additional_stop_words)

# Custom function to clean content
def clean_content(text):
    text = re.sub(r'^(subtitle|chapter|subchapter|part|subpart)\s?[a-zA-Z\d]*\s?[–—]', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'[–—]', ' ', text)  # Replace unwanted characters
    text = re.sub(r'[^a-zA-Z\s]', '', text).strip()  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Tokenize text and remove stopwords
def preprocess_and_tokenize(df, content_column):
    df['cleaned_content'] = df[content_column].apply(lambda x: clean_content(str(x)))  # Clean text
    df['tokens'] = df['cleaned_content'].apply(lambda x: [word for word in x.lower().split() if word not in stop_words and len(word) > 2])  # Tokenize and remove stopwords
    return df

# Calculate word frequencies
def calculate_word_frequencies(df):
    all_tokens = df['tokens'].explode()
    return all_tokens.value_counts()

# Get top N frequent words
def get_frequent_words(word_frequencies, top_n=50):
    return word_frequencies.head(top_n).index.tolist()

# Cluster words using Word2Vec and KMeans
def cluster_frequent_words(frequent_words, model, n_clusters):
    word_vectors = [model.wv[word] for word in frequent_words if word in model.wv]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(word_vectors)
    clusters = {i: [] for i in range(n_clusters)}
    for i, word in enumerate(frequent_words):
        if word in model.wv:  # Ensure word exists in the model
            cluster = kmeans.labels_[i]
            clusters[cluster].append(word)
    return clusters

# Prepare clusters for saving
def prepare_cluster_data(clusters):
    return [{'Cluster': cluster, 'Words': ', '.join(words[:10])} for cluster, words in clusters.items()]

# File paths
file_path_2017 = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017.xlsx'
file_path_2018 = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018.xlsx'

# Load and preprocess data
data_2017 = pd.read_excel(file_path_2017)
data_2018 = pd.read_excel(file_path_2018)

data_2017 = preprocess_and_tokenize(data_2017, 'content')
data_2018 = preprocess_and_tokenize(data_2018, 'content')

# Train Word2Vec models
model_2017 = Word2Vec(sentences=data_2017['tokens'], vector_size=100, window=5, min_count=1, workers=4)
model_2018 = Word2Vec(sentences=data_2018['tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Calculate word frequencies
frequencies_2017 = calculate_word_frequencies(data_2017)
frequencies_2018 = calculate_word_frequencies(data_2018)

# Get top frequent words
top_n = 100
frequent_words_2017 = get_frequent_words(frequencies_2017, top_n)
frequent_words_2018 = get_frequent_words(frequencies_2018, top_n)

# Perform clustering
n_clusters = 5
clusters_2017 = cluster_frequent_words(frequent_words_2017, model_2017, n_clusters)
clusters_2018 = cluster_frequent_words(frequent_words_2018, model_2018, n_clusters)

# Save frequent words and clusters to Excel
output_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\NLP_Transfer_Learning_Methods\Word2Vec\Frequent_Words_and_Clusters.xlsx'
with pd.ExcelWriter(output_path) as writer:
    # Save frequent words
    pd.DataFrame({
        'Frequent Words 2017': frequent_words_2017 + [None] * (top_n - len(frequent_words_2017)),
        'Frequent Words 2018': frequent_words_2018 + [None] * (top_n - len(frequent_words_2018))
    }).to_excel(writer, sheet_name='Frequent Words', index=False)

    # Save clusters
    clusters_df_2017 = prepare_cluster_data(clusters_2017)
    clusters_df_2018 = prepare_cluster_data(clusters_2018)
    pd.DataFrame([{'Section': '2017 Clusters'}] + clusters_df_2017 +
                 [{'Section': '2018 Clusters'}] + clusters_df_2018).to_excel(writer, sheet_name='Clusters', index=False)

print(f"Frequent words and clusters saved to {output_path}")

# Save processed data (Processed 2017, Processed 2018 tabs)
processed_data_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\NLP_Transfer_Learning_Methods\Word2Vec\cleaned_content_and_tokenization.xlsx'
with pd.ExcelWriter(processed_data_path) as writer:
    data_2017[['content', 'cleaned_content', 'tokens']].to_excel(writer, sheet_name='Processed 2017', index=False)
    data_2018[['content', 'cleaned_content', 'tokens']].to_excel(writer, sheet_name='Processed 2018', index=False)

print(f"Processed data saved to {processed_data_path}")
