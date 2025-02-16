import os
import re
import csv
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Try importing FAISS for faster similarity search
try:
    import faiss  # Facebook AI Similarity Search
    FAISS_AVAILABLE = True
except ModuleNotFoundError:
    print("⚠ FAISS is not installed. Falling back to sklearn Nearest Neighbors.")
    FAISS_AVAILABLE = False

# Load SBERT model (optimized for English legal text)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to clean headlines by removing "Sec." numbers
def clean_headlines(headlines):
    return [re.sub(r"Sec\.\s*\d+:", "", str(line)).strip() for line in headlines]

# Ensure output directory exists
output_dir = r"C:\Users\guymk\PycharmProjects\Final-Project\Output_Files"
os.makedirs(output_dir, exist_ok=True)

# File paths
file_2017_path = r"C:\Users\guymk\Downloads\Extracted_Hierarchy_Data_With_Ending_Check_2017.xlsx"
file_2018_path = r"C:\Users\guymk\Downloads\Extracted_Hierarchy_Data_With_Ending_Check_2018.xlsx"

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

# Function to generate embeddings efficiently
def get_embeddings(emb_list, batch_size=32):
    all_embeddings = []
    for i in range(0, len(emb_list), batch_size):
        batch = emb_list[i : i + batch_size]
        embeddings = model.encode(batch, convert_to_tensor=True)
        all_embeddings.append(embeddings.cpu().numpy())  # Move to CPU if processed on GPU
    return np.vstack(all_embeddings)

# Generate SBERT embeddings for all headlines
headlines17_embedding = get_embeddings(cleaned_headlines17)
headlines18_embedding = get_embeddings(cleaned_headlines18)

# Use FAISS for fast similarity search if available, otherwise fallback to sklearn
if FAISS_AVAILABLE:
    print("🔹 Using FAISS for fast similarity search")
    d = headlines17_embedding.shape[1]  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(d)  # L2 (Euclidean) similarity index
    index.add(headlines17_embedding)  # Add 2017 embeddings to the index
    distances, indices = index.search(headlines18_embedding, 1)  # Find 1 nearest neighbor
    semantic_similarity = 1 - (distances / np.max(distances))  # Normalize distances
else:
    print("🔹 Using sklearn Nearest Neighbors as fallback")
    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine").fit(headlines17_embedding)
    distances, indices = nn_model.kneighbors(headlines18_embedding)
    semantic_similarity = 1 - distances

# Similarity thresholds to test
SIMILARITY_THRESHOLDS = [0.85, 0.8, 0.75]

# Lists to store results
matched_results = {threshold: [] for threshold in SIMILARITY_THRESHOLDS}
unmatched_semantics_18_to_17 = {threshold: [] for threshold in SIMILARITY_THRESHOLDS}
unmatched_semantics_17_to_18 = {threshold: set(range(len(cleaned_headlines17))) for threshold in SIMILARITY_THRESHOLDS}

# Compare embeddings and find matches
for i, (level18, line18) in enumerate(zip(hierarchy_levels_2018, cleaned_headlines18)):
    j = indices[i][0]  # Index of closest match in 2017
    similarity_score = semantic_similarity[i][0]

    for threshold in SIMILARITY_THRESHOLDS:
        if similarity_score >= threshold:
            matched_results[threshold].append({
                "2018 Headline": line18,
                "2018 hierarchy_level": level18,
                "2017 Headline": cleaned_headlines17[j],
                "2017 hierarchy_level": hierarchy_levels_2017[j],
                "Semantic Similarity": similarity_score
            })
            unmatched_semantics_17_to_18[threshold].discard(j)  # Remove matched 2017 index
        else:
            unmatched_semantics_18_to_17[threshold].append({"2018 Headline": line18, "2018 hierarchy_level": level18})

# Convert unmatched 2017 headlines to list
for threshold in SIMILARITY_THRESHOLDS:
    unmatched_semantics_17_to_18[threshold] = [
        {"2017 Headline": cleaned_headlines17[j], "2017 hierarchy_level": hierarchy_levels_2017[j]}
        for j in unmatched_semantics_17_to_18[threshold]
    ]

# Save results to CSV for each threshold
for threshold in SIMILARITY_THRESHOLDS:
    csv_output_path = os.path.join(output_dir, f"matched_results_sbert_{int(threshold * 100)}.csv")
    with open(csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["2018 Headline", "2018 hierarchy_level", "2017 Headline", "2017 hierarchy_level", "Semantic Similarity"])
        for match in matched_results[threshold]:
            csv_writer.writerow([
                match["2018 Headline"],
                match["2018 hierarchy_level"],
                match["2017 Headline"],
                match["2017 hierarchy_level"],
                f"{match['Semantic Similarity']:.4f}"
            ])
    print(f"📂 Results for threshold {threshold} saved to: {csv_output_path}")

# Save unmatched results for each threshold
for threshold in SIMILARITY_THRESHOLDS:
    unmatched_csv_output_path = os.path.join(output_dir, f"unmatched_results_sbert_{int(threshold * 100)}.csv")
    with open(unmatched_csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Year", "Headline", "Hierarchy Level", "Unmatched Direction"])

        for entry in unmatched_semantics_18_to_17[threshold]:
            csv_writer.writerow(["2018", entry["2018 Headline"], entry["2018 hierarchy_level"], "2018 -> 2017"])

        for entry in unmatched_semantics_17_to_18[threshold]:
            csv_writer.writerow(["2017", entry["2017 Headline"], entry["2017 hierarchy_level"], "2017 -> 2018"])

    print(f"📂 Unmatched results for threshold {threshold} saved to: {unmatched_csv_output_path}")
