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

# Load headlines
with open("D:\\yifat\\Final_Project\\INTERNAL REVENUE CODE\\USCODE-2017_Headlines_text.txt", "r", encoding="utf-8") as f1:
    headlines17 = [line.strip() for line in f1.readlines()]

with open("D:\\yifat\\Final_Project\\INTERNAL REVENUE CODE\\USCODE-2018_Headlines_text.txt", "r", encoding="utf-8") as f2:
    headlines18 = [line.strip() for line in f2.readlines()]

# Clean the headlines to remove "Sec." numbers
cleaned_headlines17 = clean_headlines(headlines17)
cleaned_headlines18 = clean_headlines(headlines18)

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

# Initialize unmatched lists
unmatched_semantics_18_to_17 = []  # For unmatched headlines from 2018 to 2017
unmatched_semantics_17_to_18 = []  # For unmatched headlines from 2017 to 2018
matched_results = []  # For matched headlines with similarity scores

# Compare embeddings and text
# Compare 2018 -> 2017 (One direction)
for i, line18 in enumerate(cleaned_headlines18):  # Iterate over 2018 cleaned headlines
    matched = False

    for j, line17 in enumerate(cleaned_headlines17):  # Iterate over 2017 cleaned headlines
        semantic_score = semantic_similarity[i][j]
        levenshtein_score = levenshtein_distance(line18, line17)
        jaccard_score = jaccard_similarity(line18, line17)

        # If any metric meets the threshold, consider it a match
        if semantic_score >= SEMANTIC_THRESHOLD or levenshtein_score <= LEVENSHTEIN_THRESHOLD or jaccard_score >= JACCARD_THRESHOLD:
            matched = True
            matched_results.append({
                "2018": line18,
                "2017": line17,
                "Semantic Similarity": semantic_score,
                "Levenshtein Distance": levenshtein_score,
                "Jaccard Similarity": jaccard_score
            })

    if not matched:
        unmatched_semantics_18_to_17.append(line18)


# Compare 2017 -> 2018 (reverse direction)
for j, line17 in enumerate(cleaned_headlines17):  # Iterate over 2017 cleaned headlines
    matched = False

    for i, line18 in enumerate(cleaned_headlines18):  # Iterate over 2018 cleaned headlines
        semantic_score = semantic_similarity[i][j]
        levenshtein_score = levenshtein_distance(line17, line18)
        jaccard_score = jaccard_similarity(line17, line18)

        if semantic_score >= SEMANTIC_THRESHOLD or levenshtein_score <= LEVENSHTEIN_THRESHOLD or jaccard_score >= JACCARD_THRESHOLD:
            matched = True

    if not matched:
        unmatched_semantics_17_to_18.append(line17)

# Print unmatched results
print(f"\nNumber of lines from 2018 with no semantic match in 2017: {len(unmatched_semantics_18_to_17)}")
print("Lines from 2018 with no semantic match in 2017:")
for line in unmatched_semantics_18_to_17:
    print(f"- {line}")

print(f"\nNumber of lines from 2017 with no semantic match in 2018: {len(unmatched_semantics_17_to_18)}")
print("Lines from 2017 with no semantic match in 2018:")
for line in unmatched_semantics_17_to_18:
    print(f"- {line}")

# Print matched results
# Print matched results
print(f"\nNumber of matched lines: {len(matched_results)}")
print("Matched lines with similarity scores:")
for match in matched_results:
    print(f"- 2018: {match['2018']}\n  2017: {match['2017']}\n"
          f"  Semantic Similarity: {match['Semantic Similarity']:.2f}\n"
          f"  Levenshtein Distance: {match['Levenshtein Distance']}\n"
          f"  Jaccard Similarity: {match['Jaccard Similarity']:.2f}\n")


# Export results to files
# Define output file paths
output_dir = "D:\\yifat\\Final_Project\\Output_Files"
txt_output_path = os.path.join(output_dir, "comparison_results.txt")
csv_output_path = os.path.join(output_dir, "matched_results.csv")

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Write results to a text file
with open(txt_output_path, "w", encoding="utf-8") as txt_file:
    # Unmatched results from 2018 to 2017
    txt_file.write(f"Number of lines from 2018 with no semantic match in 2017: {len(unmatched_semantics_18_to_17)}\n")
    txt_file.write("Lines from 2018 with no semantic match in 2017:\n")
    for line in unmatched_semantics_18_to_17:
        txt_file.write(f"- {line}\n")
    txt_file.write("\n")

    # Unmatched results from 2017 to 2018
    txt_file.write(f"Number of lines from 2017 with no semantic match in 2018: {len(unmatched_semantics_17_to_18)}\n")
    txt_file.write("Lines from 2017 with no semantic match in 2018:\n")
    for line in unmatched_semantics_17_to_18:
        txt_file.write(f"- {line}\n")
    txt_file.write("\n")

    # Matched results summary
    txt_file.write(f"Number of matched lines: {len(matched_results)}\n")
    txt_file.write("Matched lines with similarity scores:\n")
    for match in matched_results:
        txt_file.write(f"- 2018: {match['2018']}\n  2017: {match['2017']}\n"
                       f"  Semantic Similarity: {match['Semantic Similarity']:.2f}\n"
                       f"  Levenshtein Distance: {match['Levenshtein Distance']}\n"
                       f"  Jaccard Similarity: {match['Jaccard Similarity']:.2f}\n")
        txt_file.write("\n")

# Write matched results to a CSV file
with open(csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header
    csv_writer.writerow(["2018 Headline", "2017 Headline", "Semantic Similarity", "Levenshtein Distance", "Jaccard Similarity"])
    # Write matched results
    for match in matched_results:
        csv_writer.writerow([match["2018"], match["2017"], f"{match['Semantic Similarity']:.2f}",
                             match["Levenshtein Distance"], f"{match['Jaccard Similarity']:.2f}"])

print(f"Results have been exported to:\n- {txt_output_path}\n- {csv_output_path}")