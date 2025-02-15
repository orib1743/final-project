import os
import re
import csv
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load the SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to clean headlines by removing "Sec." numbers
def clean_headlines(headlines):
    return [re.sub(r"Sec\.\s*\d+:", "", str(line)).strip() for line in headlines]

# Ensure output directory exists
output_dir = r"C:\Users\guymk\PycharmProjects\Final-Project\Output_Files"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# File paths
file_2017_path = r"C:\Users\guymk\Downloads\Content_2017_With_Embeddings.xlsx"
file_2018_path = r"C:\Users\guymk\Downloads\Content_2018_With_Embeddings.xlsx"

# Function to load headlines and hierarchy levels
def load_headlines_with_levels(file_path):
    df = pd.read_excel(file_path)
    hierarchy_levels = df["hierarchy_level"].dropna().tolist()
    headlines = df["hierarchy_level_name"].dropna().tolist()
    cleaned_headlines = clean_headlines(headlines)
    return hierarchy_levels, cleaned_headlines

# Load headlines
hierarchy_levels_2017, cleaned_headlines17 = load_headlines_with_levels(file_2017_path)
hierarchy_levels_2018, cleaned_headlines18 = load_headlines_with_levels(file_2018_path)

# Function to generate embeddings in batches
def get_embeddings(emb_list, batch_size=32):
    all_embeddings = []
    for i in range(0, len(emb_list), batch_size):
        batch = emb_list[i : i + batch_size]
        embeddings = model.encode(batch, convert_to_tensor=True)
        all_embeddings.append(embeddings.cpu().numpy())  # Move to CPU if processed on GPU
    return np.vstack(all_embeddings)

# Generate embeddings for all headlines
headlines17_embedding = get_embeddings(cleaned_headlines17)
headlines18_embedding = get_embeddings(cleaned_headlines18)

# Compute cosine similarity using nearest neighbors
nn_model = NearestNeighbors(n_neighbors=1, metric="cosine").fit(headlines17_embedding)
distances, indices = nn_model.kneighbors(headlines18_embedding)

# Convert distances to similarity scores (1 - distance)
semantic_similarity = 1 - distances

# Similarity threshold
SEMANTIC_THRESHOLD = 0.8

# Lists to store results
matched_results = []
unmatched_semantics_18_to_17 = []
unmatched_semantics_17_to_18 = set(range(len(cleaned_headlines17)))  # Track unmatched indices

# Compare embeddings and find matches
for i, (level18, line18) in enumerate(zip(hierarchy_levels_2018, cleaned_headlines18)):
    j = indices[i][0]  # Index of the closest match in 2017
    semantic_score = semantic_similarity[i][0]

    if semantic_score >= SEMANTIC_THRESHOLD:
        matched_results.append({
            "2018 Headline": line18,
            "2018 hierarchy_level": level18,
            "2017 Headline": cleaned_headlines17[j],
            "2017 hierarchy_level": hierarchy_levels_2017[j],
            "Semantic Similarity": semantic_score
        })
        unmatched_semantics_17_to_18.discard(j)  # Remove matched 2017 index
    else:
        unmatched_semantics_18_to_17.append({"2018 Headline": line18, "2018 hierarchy_level": level18})

# Convert unmatched 2017 headlines to list
unmatched_semantics_17_to_18 = [{"2017 Headline": cleaned_headlines17[j], "2017 hierarchy_level": hierarchy_levels_2017[j]} for j in unmatched_semantics_17_to_18]

# Print unmatched results
def print_unmatched_results(unmatched_18_to_17, unmatched_17_to_18):
    print(f"\n🔹 Headlines from 2018 with no match in 2017: {len(unmatched_18_to_17)}")
    for entry in unmatched_18_to_17[:5]:  # Print only the first 5 for brevity
        print(f"- 2018: {entry['2018 Headline']} | Hierarchy Level: {entry['2018 hierarchy_level']}")

    print(f"\n🔹 Headlines from 2017 with no match in 2018: {len(unmatched_17_to_18)}")
    for entry in unmatched_17_to_18[:5]:  # Print only the first 5 for brevity
        print(f"- 2017: {entry['2017 Headline']} | Hierarchy Level: {entry['2017 hierarchy_level']}")

print_unmatched_results(unmatched_semantics_18_to_17, unmatched_semantics_17_to_18)

# Write matched results to CSV file
csv_output_path = os.path.join(output_dir, "matched_results_embedding.csv")
with open(csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["2018 Headline", "2018 hierarchy_level", "2017 Headline", "2017 hierarchy_level", "Semantic Similarity"])
    for match in matched_results:
        csv_writer.writerow([
            match["2018 Headline"],
            match["2018 hierarchy_level"],
            match["2017 Headline"],
            match["2017 hierarchy_level"],
            f"{match['Semantic Similarity']:.4f}"
        ])

print(f"📂 Results have been saved to: {csv_output_path}")

# Write unmatched results to CSV file
unmatched_csv_output_path = os.path.join(output_dir, "unmatched_results_embedding.csv")
with open(unmatched_csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Year", "Headline", "Hierarchy Level", "Unmatched Direction"])

    for entry in unmatched_semantics_18_to_17:
        csv_writer.writerow(["2018", entry["2018 Headline"], entry["2018 hierarchy_level"], "2018 -> 2017"])

    for entry in unmatched_semantics_17_to_18:
        csv_writer.writerow(["2017", entry["2017 Headline"], entry["2017 hierarchy_level"], "2017 -> 2018"])

print(f"📂 Unmatched results have been saved to: {unmatched_csv_output_path}")
