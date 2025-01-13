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
    if pd.isna(text):  # Check if the value is NaN
        return ""  # Return an empty string for NaN values
    text = str(text).strip()  # Ensure the text is a string and strip spaces
    text = re.sub(r'^\(\w+\)\s*', '', text)  # Remove parentheses at the beginning of the paragraph
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\b(paragraph|subparagraph|section|part|subpart|chapter|subchapter|subtitle)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to get BERT embeddings
def get_embeddings(text):
    if not text:
        return None  # Return None for empty text
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model(**tokens)
        embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling
    return embeddings

# Load the matched results dataset
matched_results_path = r"C:\Users\tomersp10\Desktop\Final - Project\Final-Project\Output_Files\matched_results_with_levels.csv"
matched_results_df = pd.read_csv(matched_results_path)

# Load additional datasets
file_2017_path = r"C:\Users\tomersp10\Desktop\Final - Project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Data_With_Ending_Check_2017.xlsx"
file_2018_path = r"C:\Users\tomersp10\Desktop\Final - Project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Data_With_Ending_Check_2018.xlsx"

df_2017 = pd.read_excel(file_2017_path)
df_2018 = pd.read_excel(file_2018_path)

# Filter relevant columns
df_2017 = df_2017[['hierarchy_level_name', 'hierarchy_level', 'last_hierarchy_level_name']]
df_2018 = df_2018[['hierarchy_level_name', 'hierarchy_level', 'last_hierarchy_level_name']]

# Function to retrieve children one level below a parent title
def get_children(df, parent_title):
    # Find rows where the 'last_hierarchy_level_name' matches the parent title
    return df[df['last_hierarchy_level_name'].apply(clean_text) == clean_text(parent_title)]

# Function to compare children pairs
def compare_children_pairs(parent_2017, parent_2018, hierarchy_level):
    children_2017 = get_children(df_2017, parent_2017)
    children_2018 = get_children(df_2018, parent_2018)

    comparison_results = []
    for _, child_2017 in children_2017.iterrows():
        embedding_child_2017 = get_embeddings(child_2017['hierarchy_level_name'])
        if embedding_child_2017 is None:  # Skip if embedding is None
            continue
        for _, child_2018 in children_2018.iterrows():
            embedding_child_2018 = get_embeddings(child_2018['hierarchy_level_name'])
            if embedding_child_2018 is None:  # Skip if embedding is None
                continue

            # Remove extra brackets to avoid 3D array issue
            semantic_score = cosine_similarity(embedding_child_2017, embedding_child_2018)[0][0]

            # Store the comparison result
            comparison_results.append({
                "Parent Title 2017": parent_2017,
                "Hierarchy Level 2017": hierarchy_level,
                "Parent Title 2018": parent_2018,
                "Hierarchy Level 2018": hierarchy_level,
                "Child Title 2017": child_2017['hierarchy_level_name'],
                "Child Title 2018": child_2018['hierarchy_level_name'],
                "Child Hierarchy Level 2017": child_2017['hierarchy_level'],
                "Child Hierarchy Level 2018": child_2018['hierarchy_level'],
                "Semantic Similarity Score": semantic_score
            })
    return comparison_results


# List to store all comparison results
all_comparisons = []

# Iterate over matched parent pairs with the same hierarchy level
for _, row in matched_results_df.iterrows():
    if row['2017 hierarchy_level'] == row['2018 hierarchy_level']:
        parent_2017 = row['2017 Headline']
        parent_2018 = row['2018 Headline']
        hierarchy_level = row['2017 hierarchy_level']  # Same hierarchy level for both

        # Compare children of matched parent pairs
        comparisons = compare_children_pairs(parent_2017, parent_2018, hierarchy_level)
        all_comparisons.extend(comparisons)


# Create DataFrame and save to CSV
results_df = pd.DataFrame(all_comparisons)
output_path = r"C:\Users\tomersp10\Desktop\Final - Project\Final-Project\Output_Files\child_level_comparison_02.csv"
results_df.to_csv(output_path, index=False)

print("Results saved successfully!")