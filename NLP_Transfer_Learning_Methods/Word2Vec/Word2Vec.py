import pandas as pd
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
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

# Calculate word frequency for each document
def calculate_word_frequencies(df):
    all_tokens = df['tokens'].explode()
    word_frequencies = all_tokens.value_counts()
    return word_frequencies

# Get the most frequent words
def get_frequent_words(word_frequencies, top_n=50):
    return word_frequencies.head(top_n).index.tolist()

# Perform clustering on frequent words using Word2Vec
def cluster_frequent_words(frequent_words, model, n_clusters):
    word_vectors = [model.wv[word] for word in frequent_words if word in model.wv]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(word_vectors)
    clusters = {i: [] for i in range(n_clusters)}
    for i, word in enumerate(frequent_words):
        if word in model.wv:  # Ensure the word has a vector in the model
            cluster = kmeans.labels_[i]
            clusters[cluster].append(word)
    return clusters

# Prepare cluster data for export
def prepare_cluster_data(clusters):
    return [{'Cluster': cluster, 'Words': ', '.join(words[:10])} for cluster, words in clusters.items()]

# Load files
data_2018_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017.xlsx'
data_2017_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018.xlsx'

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

# Calculate word frequencies for each year
frequencies_2017 = calculate_word_frequencies(data_2017)
frequencies_2018 = calculate_word_frequencies(data_2018)

# Get the most frequent words
top_n = 50  # You can adjust this value
frequent_words_2017 = get_frequent_words(frequencies_2017, top_n)
frequent_words_2018 = get_frequent_words(frequencies_2018, top_n)

# Perform clustering on frequent words
n_clusters = 5  # Number of clusters
clusters_frequent_2017 = cluster_frequent_words(frequent_words_2017, model_2017, n_clusters)
clusters_frequent_2018 = cluster_frequent_words(frequent_words_2018, model_2018, n_clusters)

# Save results of frequent words and clusters to a new Excel file
frequent_words_output_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\NLP_Transfer_Learning_Methods\Frequent_Words_and_Clusters.xlsx'
with pd.ExcelWriter(frequent_words_output_path) as writer:
    # Frequent words
    pd.DataFrame({
        'Frequent Words 2017': frequent_words_2017 + [None] * (top_n - len(frequent_words_2017)),
        'Frequent Words 2018': frequent_words_2018 + [None] * (top_n - len(frequent_words_2018))
    }).to_excel(writer, sheet_name='Frequent Words', index=False)

    # Clusters of frequent words
    frequent_topics_2017 = prepare_cluster_data(clusters_frequent_2017)
    frequent_topics_2018 = prepare_cluster_data(clusters_frequent_2018)
    pd.DataFrame([{'Section': 'Frequent Topics 2017'}] + frequent_topics_2017 +
                 [{'Section': 'Frequent Topics 2018'}] + frequent_topics_2018).to_excel(writer, sheet_name='Frequent Clusters', index=False)

print(f"Frequent words and clusters saved to {frequent_words_output_path}")

# Calculate new and disappeared words
words_2017_set = set(data_2017['tokenized_words'].str.split(expand=True).stack().unique())
words_2018_set = set(data_2018['tokenized_words'].str.split(expand=True).stack().unique())

new_words = list(words_2018_set - words_2017_set)  # Words that appear in 2018 but not in 2017
disappeared_words = list(words_2017_set - words_2018_set)  # Words that appear in 2017 but not in 2018

# Align the lengths of the new and disappeared word lists
max_length = max(len(new_words), len(disappeared_words))
new_words += [None] * (max_length - len(new_words))
disappeared_words += [None] * (max_length - len(disappeared_words))

# Continue saving the original output as well
output_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\NLP_Transfer_Learning_Methods\Keyword_and_Topic_Analysis_Filtered_Cleaned.xlsx'

with pd.ExcelWriter(output_path) as writer:
    # New and disappeared words
    pd.DataFrame({'New Words in 2018': new_words, 'Disappeared Words in 2017': disappeared_words}).to_excel(
        writer, sheet_name='Keyword Changes', index=False)
    # Combine clusters from 2017 and 2018
    combined_topics = pd.DataFrame([{'Section': 'Topics 2017'}] + prepare_cluster_data(clusters_frequent_2017) +
                                   [{'Section': 'Topics 2018'}] + prepare_cluster_data(clusters_frequent_2018))
    combined_topics.to_excel(writer, sheet_name='Frequent Topics', index=False)
    # Processed text from 2018
    data_2018[['hierarchy_level_name', 'cleaned_hierarchy_level_name', 'tokenized_words']].to_excel(writer, sheet_name='Cleaned Data 2018', index=False)
    # Processed text from 2017
    data_2017[['hierarchy_level_name', 'cleaned_hierarchy_level_name', 'tokenized_words']].to_excel(writer, sheet_name='Cleaned Data 2017', index=False)

print(f"Results saved to {output_path}")