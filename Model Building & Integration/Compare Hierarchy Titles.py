import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance

# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
model = AutoModel.from_pretrained("sentence-transformers/LaBSE")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'^\(\w+\)\s*', '', text)  # Remove parentheses at the beginning of the paragraph
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\b(paragraph|subparagraph|section|part|subpart|chapter|subchapter|subtitle)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to get BERT embeddings
def get_embeddings(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model(**tokens)
        embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling
    return embeddings

# Load the matched results dataset
matched_results_path = r"C:\Users\תומר סורוז'ון\Desktop\לימודים\Final Project\Final-Project\Final-Project\Output_Files\matched_results_with_levels.csv"
matched_results_df = pd.read_csv(matched_results_path)

# Remove duplicate pairs of parent headlines
matched_results_df = matched_results_df.drop_duplicates(subset=['2017 Headline', '2018 Headline'])

# Load additional datasets
file_2017_path = r"C:\Users\תומר סורוז'ון\Desktop\לימודים\Final Project\Final-Project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Data_With_Ending_Check_2017.xlsx"
file_2018_path = r"C:\Users\תומר סורוז'ון\Desktop\לימודים\Final Project\Final-Project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Data_With_Ending_Check_2018.xlsx"

df_2017 = pd.read_excel(file_2017_path)
df_2018 = pd.read_excel(file_2018_path)

# Filter relevant columns
df_2017 = df_2017[['hierarchy_level_name', 'hierarchy_level', 'last_hierarchy_level_name']]
df_2018 = df_2018[['hierarchy_level_name', 'hierarchy_level', 'last_hierarchy_level_name']]

def get_best_matches(grouped_matched_results):
    """ Select the best match for each 2017 headline based on the highest semantic similarity. """
    best_matches = []

    for _, group in grouped_matched_results.groupby('2017 Headline'):
        best_match = None
        highest_score = -1

        for _, row in group.iterrows():
            parent_2017 = row['2017 Headline']
            parent_2018 = row['2018 Headline']

            # Clean and get embeddings
            embedding_2017 = get_embeddings(clean_text(parent_2017))
            embedding_2018 = get_embeddings(clean_text(parent_2018))

            # Calculate semantic similarity score
            semantic_score = cosine_similarity(embedding_2017, embedding_2018)[0][0]

            if semantic_score > highest_score:
                highest_score = semantic_score
                best_match = row

        if best_match is not None:
            best_matches.append(best_match)

    return pd.DataFrame(best_matches)

# Function to compare child levels with hierarchy after selecting the best match
def compare_child_levels_with_hierarchy(parent_2017, parent_2018, hierarchy_level_2017, hierarchy_level_2018):
    clean_parent_2017 = clean_text(parent_2017)
    clean_parent_2018 = clean_text(parent_2018)

    children_2017 = df_2017[df_2017['last_hierarchy_level_name'] == parent_2017]
    children_2018 = df_2018[df_2018['last_hierarchy_level_name'] == parent_2018]

    total_children_2017 = len(children_2017)
    total_children_2018 = len(children_2018)

    max_possible_matches = min(total_children_2017, total_children_2018)

    if total_children_2017 == 0 or total_children_2018 == 0:
        return {
            "Parent Title 2017": clean_parent_2017,
            "Hierarchy Level 2017": hierarchy_level_2017,
            "Total Children 2017": total_children_2017,
            "Parent Title 2018": clean_parent_2018,
            "Hierarchy Level 2018": hierarchy_level_2018,
            "Total Children 2018": total_children_2018,
            "Semantic Matches": 0,
            "Semantic Similarity Avg": 0,
            "Levenshtein_Matches": 0,
            "Levenshtein Distance Avg": 0,
            "Jaccard_Matches": 0,
            "Jaccard Similarity Avg": 0,
            "Matching Children": 0,
            "Matching Children Same Level": 0
        }

    # Initialize counters and lists for matches
    matching_children = 0
    same_hierarchy_count = 0
    semantic_similarities = []
    levenshtein_distances = []
    jaccard_similarities = []

    # Shared matched indices sets across both comparisons
    matched_indices_semantic = set()
    matched_indices_levenshtein = set()
    matched_indices_jaccard = set()
    matched_indices_all = set()

    def compare_children(children_a, children_b):
        nonlocal matching_children, same_hierarchy_count, matched_indices_semantic, matched_indices_levenshtein, matched_indices_jaccard, matched_indices_all
        for i, child_a in children_a.iterrows():
            clean_child_a = clean_text(child_a['hierarchy_level_name'])
            embedding_child_a = get_embeddings(clean_child_a)

            for j, child_b in children_b.iterrows():
                if j in matched_indices_all:  # Ensure no double-matching
                    continue

                clean_child_b = clean_text(child_b['hierarchy_level_name'])
                embedding_child_b = get_embeddings(clean_child_b)

                semantic_score = cosine_similarity(embedding_child_a, embedding_child_b)[0][0]
                levenshtein_score = levenshtein_distance(clean_child_a, clean_child_b)
                jaccard_score = len(set(clean_child_a.split()).intersection(set(clean_child_b.split()))) / len(set(clean_child_a.split()).union(set(clean_child_b.split())))

                # Add to sets only if thresholds are met
                if semantic_score >= 0.8:
                    matched_indices_semantic.add(j)
                    semantic_similarities.append(semantic_score)
                if levenshtein_score <= 10:
                    matched_indices_levenshtein.add(j)
                    levenshtein_distances.append(levenshtein_score)
                if jaccard_score >= 0.5:
                    matched_indices_jaccard.add(j)
                    jaccard_similarities.append(jaccard_score)

                # Count as full match only if all conditions are met
                if semantic_score >= 0.8 and levenshtein_score <= 10 and jaccard_score >= 0.5:
                    matched_indices_all.add(j)
                    matching_children += 1

                    if ".".join(child_a['hierarchy_level'].split(".")[:-1]) == ".".join(child_b['hierarchy_level'].split(".")[:-1]):
                        same_hierarchy_count += 1

                    # Break to avoid multiple matches for the same child
                    break

                if matching_children >= max_possible_matches:
                    break

    # Compare children in both directions
    compare_children(children_2017, children_2018)
    compare_children(children_2018, children_2017)

    avg_semantic_similarity = (sum(semantic_similarities) / len(matched_indices_semantic)) if len(matched_indices_semantic) > 0 else 0
    avg_levenshtein_distance = (sum(levenshtein_distances) / len(matched_indices_levenshtein)) if len(matched_indices_levenshtein) > 0 else 0
    avg_jaccard_similarity = (sum(jaccard_similarities) / len(matched_indices_jaccard)) if len(matched_indices_jaccard) > 0 else 0

    return {
        "Parent Title 2017": clean_parent_2017,
        "Hierarchy Level 2017": hierarchy_level_2017,
        "Total Children 2017": total_children_2017,
        "Parent Title 2018": clean_parent_2018,
        "Hierarchy Level 2018": hierarchy_level_2018,
        "Total Children 2018": total_children_2018,
        "Semantic Matches": len(matched_indices_semantic),
        "Semantic Similarity Avg": avg_semantic_similarity,
        "Levenshtein_Matches": len(matched_indices_levenshtein),
        "Levenshtein Distance Avg": avg_levenshtein_distance,
        "Jaccard_Matches": len(matched_indices_jaccard),
        "Jaccard Similarity Avg": avg_jaccard_similarity,
        "Matching Children": matching_children,
        "Matching Children Same Level": same_hierarchy_count
    }



# Iterate over matched results and run comparisons
best_matched_results = get_best_matches(matched_results_df)

results = []
for _, row in best_matched_results.iterrows():
    if row['2017 hierarchy_level'] == row['2018 hierarchy_level']:
        parent_2017 = row['2017 Headline']
        parent_2018 = row['2018 Headline']
        hierarchy_level_2017 = row['2017 hierarchy_level']
        hierarchy_level_2018 = row['2018 hierarchy_level']

        comparison_result = compare_child_levels_with_hierarchy(parent_2017, parent_2018, hierarchy_level_2017, hierarchy_level_2018)
        results.append(comparison_result)

results_df = pd.DataFrame(results)
results_df.to_csv(r"C:\Users\תומר סורוז'ון\Desktop\לימודים\Final Project\Final-Project\Final-Project\Output_Files\child_level_comparison.csv", index=False)

print("Results saved successfully!")
