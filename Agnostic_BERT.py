from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# Load the LaBSE model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE", do_lower_case=False)
labse_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")

# Set device to CUDA if available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labse_model = labse_model.to(device)

# Load headlines
with open("D:\\יפעת\\לימודים\\שנה ג\\פרויקט גמר\\קוד פייתון\\USCODE-2017_Headlines_text.txt", "r", encoding="utf-8") as f1:
    headlines17 = [line.strip() for line in f1.readlines()]

with open("D:\\יפעת\\לימודים\\שנה ג\\פרויקט גמר\\קוד פייתון\\USCODE-2018_Headlines_text.txt", "r", encoding="utf-8") as f2:
    headlines18 = [line.strip() for line in f2.readlines()]

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
headlines17_embedding = get_embeddings(headlines17)
headlines18_embedding = get_embeddings(headlines18)

# Define thresholds
SEMANTIC_THRESHOLD = 0.8


# Compute semantic similarity matrix
semantic_similarity = cosine_similarity(headlines18_embedding, headlines17_embedding)

# Initialize unmatched lists
unmatched_semantics = []


# Compare embeddings and text
for i, line18 in enumerate(headlines18):  # Iterate over 2018 headlines (text)
    matched_semantics = False

    for j, line17 in enumerate(headlines17):  # Iterate over 2017 headlines (text)
        # Check semantic similarity using embeddings
        if semantic_similarity[i][j] >= SEMANTIC_THRESHOLD:
            matched_semantics = True


    # If no match found, add to unmatched lists
    if not matched_semantics:
        unmatched_semantics.append(line18)

# Print differences

print("\nLines with no semantic match (Semantic Difference):")
for line in unmatched_semantics:
    print(f"- {line}")
