import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from Levenshtein import distance as levenshtein_distance
import re

# Load the LaBSE model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE", do_lower_case=False)
labse_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
labse_model.eval()

# Set device to CUDA if available, else use CPU
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
labse_model = labse_model.to(device)

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):  # Check if the input is not a string
        return ""
    return re.sub(r"\s+", " ", text).strip()


# Function to extract embeddings
# Function to extract embeddings
def get_embeddings(texts, tokenizer, model, device):
    tokenized = tokenizer(
        texts,
        add_special_tokens=True,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(
            input_ids=tokenized['input_ids'].to(device),
            attention_mask=tokenized['attention_mask'].to(device)
        )
        embeddings = outputs[1].cpu().numpy()
    return normalize(embeddings)


# Load the datasets
file_2017_path = "C:\\Users\\תומר סורוז'ון\\Desktop\\לימודים\\Final Project\\Final-Project\\Final-Project\\INTERNAL REVENUE CODE\\Extracted_Data\\Extracted_Hierarchy_Data_With_Paragraph_2017.xlsx"
file_2018_path = "C:\\Users\\תומר סורוז'ון\\Desktop\\לימודים\\Final Project\\Final-Project\\Final-Project\\INTERNAL REVENUE CODE\\Extracted_Data\\Extracted_Hierarchy_Data_With_Paragraph_2018.xlsx"

data_2017 = pd.read_excel(file_2017_path)
data_2018 = pd.read_excel(file_2018_path)

# Extract all level 7 paragraphs
data_2017_paragraphs = data_2017[data_2017['hierarchy_level'] == "7.Paragraph"][['hierarchy_level_name', 'content']]
data_2018_paragraphs = data_2018[data_2018['hierarchy_level'] == "7.Paragraph"][['hierarchy_level_name', 'content']]

data_2017_paragraphs = data_2017_paragraphs.head(100)  # השתמש רק ב-100 הפסקאות הראשונות
data_2018_paragraphs = data_2018_paragraphs.head(100)


# Prepare results containers
semantic_results_2017_to_2018 = []
semantic_results_2018_to_2017 = []
lyrical_results_2017_to_2018 = []
lyrical_results_2018_to_2017 = []

# Comparison: 2017 to 2018
for _, row_2017 in data_2017_paragraphs.iterrows():
    title_2017, paragraph_2017 = row_2017['hierarchy_level_name'], clean_text(row_2017['content'])
    for _, row_2018 in data_2018_paragraphs.iterrows():
        title_2018, paragraph_2018 = row_2018['hierarchy_level_name'], clean_text(row_2018['content'])

        # Semantic similarity
        embeddings_2017 = get_embeddings([paragraph_2017], tokenizer, labse_model, device)
        embeddings_2018 = get_embeddings([paragraph_2018], tokenizer, labse_model, device)
        semantic_score = cosine_similarity(embeddings_2017, embeddings_2018)[0][0]

        semantic_results_2017_to_2018.append({
            "Title_2017": title_2017,
            "Paragraph_2017": paragraph_2017,
            "Title_2018": title_2018,
            "Paragraph_2018": paragraph_2018,
            "Semantic_Similarity": semantic_score
        })

        # Lyrical similarity
        levenshtein_score = levenshtein_distance(paragraph_2017, paragraph_2018)
        lyrical_results_2017_to_2018.append({
            "Title_2017": title_2017,
            "Paragraph_2017": paragraph_2017,
            "Title_2018": title_2018,
            "Paragraph_2018": paragraph_2018,
            "Levenshtein_Distance": levenshtein_score
        })

# Comparison: 2018 to 2017
for _, row_2018 in data_2018_paragraphs.iterrows():
    title_2018, paragraph_2018 = row_2018['hierarchy_level_name'], clean_text(row_2018['content'])
    for _, row_2017 in data_2017_paragraphs.iterrows():
        title_2017, paragraph_2017 = row_2017['hierarchy_level_name'], clean_text(row_2017['content'])

        # Semantic similarity
        embeddings_2018 = get_embeddings([paragraph_2018], tokenizer, labse_model, device)
        embeddings_2017 = get_embeddings([paragraph_2017], tokenizer, labse_model, device)
        semantic_score = cosine_similarity(embeddings_2018, embeddings_2017)[0][0]

        semantic_results_2018_to_2017.append({
            "Title_2018": title_2018,
            "Paragraph_2018": paragraph_2018,
            "Title_2017": title_2017,
            "Paragraph_2017": paragraph_2017,
            "Semantic_Similarity": semantic_score
        })

        # Lyrical similarity
        levenshtein_score = levenshtein_distance(paragraph_2018, paragraph_2017)
        lyrical_results_2018_to_2017.append({
            "Title_2018": title_2018,
            "Paragraph_2018": paragraph_2018,
            "Title_2017": title_2017,
            "Paragraph_2017": paragraph_2017,
            "Levenshtein_Distance": levenshtein_score
        })

# Convert results to DataFrames and export
pd.DataFrame(semantic_results_2017_to_2018).to_csv(r"C:\\Users\\תומר סורוז'ון\\Desktop\\Semantic_2017_to_2018.csv", index=False)
pd.DataFrame(lyrical_results_2017_to_2018).to_csv(r"C:\\Users\\תומר סורוז'ון\\Desktop\\Lyrical_2017_to_2018.csv", index=False)
pd.DataFrame(semantic_results_2018_to_2017).to_csv(r"C:\\Users\\תומר סורוז'ון\\Desktop\\Semantic_2018_to_2017.csv", index=False)
pd.DataFrame(lyrical_results_2018_to_2017).to_csv(r"C:\\Users\\תומר סורוז'ון\\Desktop\\Lyrical_2018_to_2017.csv", index=False)

print("Comparison complete. Results saved for both directions.")

