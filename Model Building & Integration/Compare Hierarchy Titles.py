import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
from Levenshtein import distance as levenshtein_distance

# Load the LaBSE model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE", do_lower_case=False)
labse_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labse_model = labse_model.to(device)

# Function to compute sentence embeddings
def get_embeddings(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = labse_model(**inputs).pooler_output
    return normalize(embeddings.cpu().numpy())

# Function to compute Jaccard similarity
def jaccard_similarity(line1, line2):
    set1 = set(line1.split())
    set2 = set(line2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if len(union) > 0 else 0.0

# Load the data
file_2017 = '/mnt/data/Extracted_Hierarchy_Data_With_Ending_Check_2017.xlsx'
file_2018 = '/mnt/data/Extracted_Hierarchy_Data_With_Ending_Check_2018.xlsx'

data_2017 = pd.read_excel(file_2017)
data_2018 = pd.read_excel(file_2018)

# Define thresholds
SEMANTIC_THRESHOLD = 0.8
LEVENSHTEIN_THRESHOLD = 10
JACCARD_THRESHOLD = 0.5

# All hierarchy levels
hierarchy_levels = [
    "1.Subtitle", "2.Chapter", "3.Subchapter",
    "4.Part", "5.Subpart"
]

results = []

for current_level in hierarchy_levels:
    # Filter titles at the current hierarchy level
    data_2017_level = data_2017[data_2017['hierarchy_level'] == current_level]
    data_2018_level = data_2018[data_2018['hierarchy_level'] == current_level]

    # Compute embeddings and similarity matrix
    embeddings_2017 = get_embeddings(data_2017_level['hierarchy_level_name'].tolist())
    embeddings_2018 = get_embeddings(data_2018_level['hierarchy_level_name'].tolist())
    similarity_matrix = cosine_similarity(embeddings_2018, embeddings_2017)

    for i, row_2018 in data_2018_level.iterrows():
        for j, row_2017 in data_2017_level.iterrows():
            semantic_similarity = similarity_matrix[i, j]
            levenshtein_score = levenshtein_distance(row_2018['hierarchy_level_name'], row_2017['hierarchy_level_name'])
            jaccard_score = jaccard_similarity(row_2018['hierarchy_level_name'], row_2017['hierarchy_level_name'])

            # Check if similarity meets thresholds
            if (semantic_similarity >= SEMANTIC_THRESHOLD or
                levenshtein_score <= LEVENSHTEIN_THRESHOLD or
                jaccard_score >= JACCARD_THRESHOLD):

                # Compare all lower levels for matching titles
                for lower_level in hierarchy_levels[hierarchy_levels.index(current_level)+1:]:
                    sub_2018 = data_2018[
                        (data_2018['last_hierarchy_level'] == row_2018['hierarchy_level']) &
                        (data_2018['hierarchy_level'] == lower_level)
                    ]
                    sub_2017 = data_2017[
                        (data_2017['last_hierarchy_level'] == row_2017['hierarchy_level']) &
                        (data_2017['hierarchy_level'] == lower_level)
                    ]

                    # Cartesian product for sublevels
                    cartesian_product = pd.DataFrame(
                        list(product(sub_2018['hierarchy_level_name'], sub_2017['hierarchy_level_name'])),
                        columns=['Sublevel_2018', 'Sublevel_2017']
                    )

                    # Save results
                    results.append({
                        'Current_Level': current_level,
                        'Title_2018': row_2018['hierarchy_level_name'],
                        'Title_2017': row_2017['hierarchy_level_name'],
                        'Semantic_Similarity': semantic_similarity,
                        'Levenshtein_Distance': levenshtein_score,
                        'Jaccard_Similarity': jaccard_score,
                        'Sublevel_Comparison': cartesian_product
                    })

# Export results
output_path = '/mnt/data/full_hierarchy_comparison.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for index, result in enumerate(results):
        sheet_name = f"{result['Current_Level']}_Match_{index+1}"
        result['Sublevel_Comparison'].to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Results exported to {output_path}")
