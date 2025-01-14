import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import re
from Levenshtein import distance as levenshtein_distance
import os
import csv

# Load the LaBSE model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE", do_lower_case=False)
labse_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")

# Set device to CUDA if available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labse_model = labse_model.to(device)

# Function to clean headlines by removing "Sec." numbers
def clean_headlines(headlines):
    return [re.sub(r"Sec\.\s*\d+:", "", line).strip() for line in headlines]

# Paths to the Excel files
file_2017_path = r"D:\yifat\Final_Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017_fixed.xlsx"
file_2018_path = r"D:\yifat\Final_Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018_fixed.xlsx"

# Update the load function to include hierarchy levels and headlines
def load_headlines_with_levels(file_path):
    df = pd.read_excel(file_path)
    hierarchy_levels = df["hierarchy_level"].dropna().tolist()
    headlines = df["hierarchy_level_name"].dropna().tolist()
    cleaned_headlines = clean_headlines(headlines)
    return hierarchy_levels, cleaned_headlines

# Load and clean the headlines with hierarchy levels
hierarchy_levels_2017, cleaned_headlines17 = load_headlines_with_levels(file_2017_path)
hierarchy_levels_2018, cleaned_headlines18 = load_headlines_with_levels(file_2018_path)


# Sentence embedding function
def get_embeddings(emb_list):
    tok_res = tokenizer(emb_list, add_special_tokens=True, padding='max_length', max_length=64, truncation=True, return_tensors="pt")
    input_ids = tok_res['input_ids'].to(device)
    token_type_ids = tok_res['token_type_ids'].to(device)
    attention_mask = tok_res['attention_mask'].to(device)

    with torch.no_grad():
        output = labse_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        embeddings = output[1].cpu().numpy()
        embeddings = normalize(embeddings)
    return embeddings

# Compute embeddings
headlines17_embedding = get_embeddings(cleaned_headlines17)
headlines18_embedding = get_embeddings(cleaned_headlines18)

# Define thresholds
SEMANTIC_THRESHOLD = 0.8
LEVENSHTEIN_THRESHOLD = 10  # Maximum allowable edit distance
JACCARD_THRESHOLD = 0.5    # Minimum Jaccard similarity

# Compute semantic similarity matrix
semantic_similarity = cosine_similarity(headlines18_embedding, headlines17_embedding)

# Function for Jaccard Similarity
def jaccard_similarity(line1, line2):
    set1 = set(line1.split())  # converting title 1 to a group of words
    set2 = set(line2.split())  # converting title 2 to a group of words
    intersection = set1.intersection(set2)  # common words
    union = set1.union(set2)  # union the words
    if len(union) == 0:  # check for empty union string
        return 0.0  # return 0, if so
    return len(intersection) / len(union)  # Calc Jaccard Similarity

# Initialize results lists
matched_results = []  # Matched results with all required data
unmatched_semantics_18_to_17 = []  # Unmatched from 2018 to 2017
unmatched_semantics_17_to_18 = []  # Unmatched from 2017 to 2018


# Compare embeddings and text
# Compare embeddings and generate results
for i, (level18, line18) in enumerate(zip(hierarchy_levels_2018, cleaned_headlines18)):
    matched = False

    for j, (level17, line17) in enumerate(zip(hierarchy_levels_2017, cleaned_headlines17)):
        semantic_score = semantic_similarity[i][j]
        levenshtein_score = levenshtein_distance(line18, line17)
        jaccard_score = jaccard_similarity(line18, line17)

        if semantic_score >= SEMANTIC_THRESHOLD or levenshtein_score <= LEVENSHTEIN_THRESHOLD or jaccard_score >= JACCARD_THRESHOLD:
            matched = True
            matched_results.append({
                "2018 Headline": line18,
                "2018 hierarchy_level": level18,
                "2017 Headline": line17,
                "2017 hierarchy_level": level17,
                "Semantic Similarity": semantic_score,
                "Levenshtein Distance": levenshtein_score,
                "Jaccard Similarity": jaccard_score,
            })

    if not matched:
        unmatched_semantics_18_to_17.append({"2018 Headline": line18, "2018 hierarchy_level": level18})

for j, (level17, line17) in enumerate(zip(hierarchy_levels_2017, cleaned_headlines17)):
    matched = False

    for i, (level18, line18) in enumerate(zip(hierarchy_levels_2018, cleaned_headlines18)):
        semantic_score = semantic_similarity[i][j]
        levenshtein_score = levenshtein_distance(line17, line18)
        jaccard_score = jaccard_similarity(line17, line18)

        if semantic_score >= SEMANTIC_THRESHOLD or levenshtein_score <= LEVENSHTEIN_THRESHOLD or jaccard_score >= JACCARD_THRESHOLD:
            matched = True

    if not matched:
        unmatched_semantics_17_to_18.append({"2017 Headline": line17, "2017 hierarchy_level": level17})


# Print unmatched results
def print_unmatched_results(unmatched_18_to_17, unmatched_17_to_18):
    print(f"\nNumber of lines from 2018 with no semantic match in 2017: {len(unmatched_18_to_17)}")
    print("Lines from 2018 with no semantic match in 2017:")
    for entry in unmatched_18_to_17:
        print(f"- 2018 Headline: {entry['2018 Headline']}\n  2018 hierarchy_level: {entry['2018 hierarchy_level']}\n")

    print(f"\nNumber of lines from 2017 with no semantic match in 2018: {len(unmatched_17_to_18)}")
    print("Lines from 2017 with no semantic match in 2018:")
    for entry in unmatched_17_to_18:
        print(f"- 2017 Headline: {entry['2017 Headline']}\n  2017 hierarchy_level: {entry['2017 hierarchy_level']}\n")

print_unmatched_results(unmatched_semantics_18_to_17, unmatched_semantics_17_to_18)

# Write results to CSV
output_dir = r"D:\yifat\Final_Project\Output_Files"
csv_output_path = os.path.join(output_dir, "matched_results_with_levels.csv")
with open(csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["2018 Headline", "2018 hierarchy_level", "2017 Headline", "2017 hierarchy_level",
                          "Semantic Similarity", "Levenshtein Distance", "Jaccard Similarity"])
    for match in matched_results:
        csv_writer.writerow([
            match["2018 Headline"],
            match["2018 hierarchy_level"],
            match["2017 Headline"],
            match["2017 hierarchy_level"],
            f"{match['Semantic Similarity']:.2f}",
            match["Levenshtein Distance"],
            f"{match['Jaccard Similarity']:.2f}"
        ])

print(f"Results have been exported to: {csv_output_path}")

# Define output path for unmatched results
unmatched_csv_output_path = os.path.join(output_dir, "unmatched_results.csv")

# Write unmatched results to a CSV file
with open(unmatched_csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header for unmatched results
    csv_writer.writerow(["Year", "Headline", "Hierarchy Level", "Unmatched Direction"])

    # Write unmatched results from 2018 to 2017
    for entry in unmatched_semantics_18_to_17:
        csv_writer.writerow([
            "2018",
            entry["2018 Headline"],
            entry["2018 hierarchy_level"],
            "2018 -> 2017"
        ])

    # Write unmatched results from 2017 to 2018
    for entry in unmatched_semantics_17_to_18:
        csv_writer.writerow([
            "2017",
            entry["2017 Headline"],
            entry["2017 hierarchy_level"],
            "2017 -> 2018"
        ])

print(f"Unmatched results have been exported to: {unmatched_csv_output_path}")

