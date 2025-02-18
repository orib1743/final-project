import os
import pandas as pd
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from wordcloud import WordCloud

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

# Prepare clusters for saving
def prepare_cluster_data(clusters):
    return [{'Cluster': cluster, 'Words': ', '.join(words[:10]), 'Size': len(words)} for cluster, words in clusters.items()]

# Visualize and save word clouds for clusters
def visualize_and_save_clusters(clusters, year, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for cluster_id, words in clusters.items():
        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(' '.join(words))

        # Save the word cloud as an image file
        file_name = os.path.join(output_folder, f"Year_{year}_Cluster_{cluster_id}.png")
        wordcloud.to_file(file_name)
        #print(f"Word cloud saved: {file_name}")

# File paths
file_path_2017 = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017_fixed.xlsx'
file_path_2018 = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018_fixed.xlsx'

# Load and preprocess data
data_2017 = pd.read_excel(file_path_2017)
data_2018 = pd.read_excel(file_path_2018)

data_2017 = preprocess_and_tokenize(data_2017, 'content')
data_2018 = preprocess_and_tokenize(data_2018, 'content')

# Train Word2Vec models
model_2017 = Word2Vec(sentences=data_2017['tokens'], vector_size=100, window=5, min_count=1, workers=4)
model_2018 = Word2Vec(sentences=data_2018['tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Perform clustering
top_n = 100
frequent_words_2017 = data_2017['tokens'].explode().value_counts().head(top_n).index.tolist()
frequent_words_2018 = data_2018['tokens'].explode().value_counts().head(top_n).index.tolist()

def cluster_frequent_words(frequent_words, model, n_clusters):
    word_vectors = [model.wv[word] for word in frequent_words if word in model.wv]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(word_vectors)
    clusters = {i: [] for i in range(n_clusters)}
    for i, word in enumerate(frequent_words):
        if word in model.wv:  # Ensure word exists in the model
            cluster = kmeans.labels_[i]
            clusters[cluster].append(word)
    return clusters

n_clusters = 5
clusters_2017 = cluster_frequent_words(frequent_words_2017, model_2017, n_clusters)
clusters_2018 = cluster_frequent_words(frequent_words_2018, model_2018, n_clusters)

# Specify output folder for visualizations
output_folder = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\NLP_Transfer_Learning_Methods\Word2Vec\Visualizations'

# Save word clouds for both years
visualize_and_save_clusters(clusters_2017, year=2017, output_folder=output_folder)
visualize_and_save_clusters(clusters_2018, year=2018, output_folder=output_folder)

# Compare cluster sizes
cluster_sizes_2017 = {cluster: len(words) for cluster, words in clusters_2017.items()}
cluster_sizes_2018 = {cluster: len(words) for cluster, words in clusters_2018.items()}

# Combine cluster data into a single tab
clusters_2017_df = pd.DataFrame(prepare_cluster_data(clusters_2017))
clusters_2017_df['Year'] = 2017

clusters_2018_df = pd.DataFrame(prepare_cluster_data(clusters_2018))
clusters_2018_df['Year'] = 2018

cluster_comparison_df = pd.DataFrame({
    'Cluster ID': list(cluster_sizes_2017.keys()),
    '2017 Size': list(cluster_sizes_2017.values()),
    '2018 Size': list(cluster_sizes_2018.values())
})

# Save combined data to Excel
output_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\NLP_Transfer_Learning_Methods\Word2Vec\Combined_Output.xlsx'
with pd.ExcelWriter(output_path) as writer:
    # Frequent words tab first
    pd.DataFrame({'Frequent Words 2017': frequent_words_2017, 'Frequent Words 2018': frequent_words_2018}).to_excel(writer, sheet_name='Frequent Words', index=False)

    # Combined clusters and comparison next
    clusters_combined = pd.concat([clusters_2017_df, clusters_2018_df])
    clusters_combined.to_excel(writer, sheet_name='Cluster Analysis', index=False)
    cluster_comparison_df.to_excel(writer, sheet_name='Cluster Analysis', startrow=len(clusters_combined) + 2, index=False)

    # Processed data tabs
    data_2017[['content', 'cleaned_content', 'tokens']].to_excel(writer, sheet_name='Processed 2017', index=False)
    data_2018[['content', 'cleaned_content', 'tokens']].to_excel(writer, sheet_name='Processed 2018', index=False)

print(f"Processed data, clusters, and comparisons saved to {output_path}")
