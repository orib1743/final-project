import pandas as pd
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define a set of stopwords
stop_words = set(stopwords.words('english'))

# Add additional stopwords for removal
additional_stop_words = {'title', 'section', 'pub', 'repealed', 'part,', 'chapter', 'subpart', 'subchapter', 'pub', 'subtitle', 'div'}
stop_words.update(additional_stop_words)  # Update the stopwords set

# Custom cleaning function
def clean_hierarchy_name_preserve_all_spaces(row):
    name = row['hierarchy_level_name']
    name = re.sub(r'^(subtitle|chapter|subchapter|part|subpart)\s?[a-zA-Z\d]*\s?[–—]', '', name, flags=re.IGNORECASE).strip()  # Remove titles
    name = re.sub(r'[–—]', ' ', name)  # Replace unwanted characters while preserving spaces
    name = re.sub(r'[^a-zA-Z\s]', '', name).strip()  # Remove non-alphabetic characters
    name = re.sub(r'\s+', ' ', name).strip()  # Remove extra spaces
    return name

# Text processing and tokenization function
def preprocess_and_tokenize(df, content_column):
    df['clean_content'] = df.apply(clean_hierarchy_name_preserve_all_spaces, axis=1)  # Clean text
    df['tokens'] = df['clean_content'].apply(lambda x: [word for word in x.lower().split() if word not in stop_words and len(word) > 2])  # Tokenize text
    return df

# Load files
data_2018_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Data_2018_corrected_v4_headlines.xlsx'
data_2017_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Data_With_Word_Count_2017.xlsx'

data_2018 = pd.read_excel(data_2018_path)
data_2017 = pd.read_excel(data_2017_path)

# Clean the text and save the cleaned text in a new column
data_2018['cleaned_hierarchy_level_name'] = data_2018.apply(clean_hierarchy_name_preserve_all_spaces, axis=1)
data_2017['cleaned_hierarchy_level_name'] = data_2017.apply(clean_hierarchy_name_preserve_all_spaces, axis=1)

# Process and tokenize the data
data_2018 = preprocess_and_tokenize(data_2018, 'hierarchy_level_name')
data_2017 = preprocess_and_tokenize(data_2017, 'hierarchy_level_name')

# Save text after tokenization
data_2018['tokenized_words'] = data_2018['tokens'].apply(lambda x: ' '.join(x))
data_2017['tokenized_words'] = data_2017['tokens'].apply(lambda x: ' '.join(x))

# Train Word2Vec models on tokens from each year
model_2017 = Word2Vec(sentences=data_2017['tokens'], vector_size=100, window=5, min_count=1, workers=4)
model_2018 = Word2Vec(sentences=data_2018['tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Calculate new and disappeared words
words_2017_set = set(data_2017['tokenized_words'].str.split(expand=True).stack().unique())
words_2018_set = set(data_2018['tokenized_words'].str.split(expand=True).stack().unique())

print("Sample Words 2017:", list(words_2017_set)[:10])  # Sample words from 2017
print("Sample Words 2018:", list(words_2018_set)[:10])  # Sample words from 2018

# Calculate new and disappeared words
new_words = list(words_2018_set - words_2017_set)  # Words that appear in 2018 but not in 2017
disappeared_words = list(words_2017_set - words_2018_set)  # Words that appear in 2017 but not in 2018

print("New Words:", new_words[:10])
print("Disappeared Words:", disappeared_words[:10])

# Dimensionality reduction for visualization and clustering
pca_2017 = PCA(n_components=10).fit_transform(model_2017.wv.vectors)
pca_2018 = PCA(n_components=10).fit_transform(model_2018.wv.vectors)

# Perform clustering with KMeans
n_clusters = 5
kmeans_2017 = KMeans(n_clusters=n_clusters, random_state=42).fit(pca_2017)
kmeans_2018 = KMeans(n_clusters=n_clusters, random_state=42).fit(pca_2018)

# Extract words for each cluster
def get_cluster_words(model, kmeans, n_clusters):
    clusters = {i: [] for i in range(n_clusters)}
    for i, word in enumerate(model.wv.index_to_key):
        cluster = kmeans.labels_[i]
        clusters[cluster].append(word)
    return clusters

clusters_2017 = get_cluster_words(model_2017, kmeans_2017, n_clusters)
clusters_2018 = get_cluster_words(model_2018, kmeans_2018, n_clusters)

# Prepare cluster data for export
def prepare_cluster_data(clusters):
    return [{'Cluster': cluster, 'Words': ', '.join(words[:10])} for cluster, words in clusters.items()]

topics_2017 = prepare_cluster_data(clusters_2017)
topics_2018 = prepare_cluster_data(clusters_2018)

# Align the lengths of the new and disappeared word lists
max_length = max(len(new_words), len(disappeared_words))
new_words += [None] * (max_length - len(new_words))
disappeared_words += [None] * (max_length - len(disappeared_words))

# Save results to an Excel file
output_path = 'Keyword_and_Topic_Analysis_Filtered_Cleaned.xlsx'
with pd.ExcelWriter(output_path) as writer:
    # New and disappeared words
    pd.DataFrame({'New Words in 2018': new_words, 'Disappeared Words in 2017': disappeared_words}).to_excel(
        writer, sheet_name='Keyword Changes', index=False)
    # Combine clusters from 2017 and 2018
    combined_topics = pd.DataFrame([{'Section': 'Topics 2017'}] + topics_2017 + [{'Section': 'Topics 2018'}] + topics_2018)
    combined_topics.to_excel(writer, sheet_name='Topics Combined', index=False)
    # Processed text from 2018
    data_2018[['hierarchy_level_name', 'cleaned_hierarchy_level_name', 'tokenized_words']].to_excel(writer, sheet_name='Cleaned Data 2018', index=False)
    # Processed text from 2017
    data_2017[['hierarchy_level_name', 'cleaned_hierarchy_level_name', 'tokenized_words']].to_excel(writer, sheet_name='Cleaned Data 2017', index=False)

print(f"Results saved to {output_path}")
