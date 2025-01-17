import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# File paths for 2017 and 2018 data
data_path_2017 = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017_fixed.xlsx'
data_path_2018 = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018_fixed.xlsx'


# Function to clean hierarchy names
def clean_hierarchy_name_preserve_all_spaces(row):
    name = row['hierarchy_level_name']

    # Standardize dashes
    name = re.sub(r'[–—]', '-', name)

    # Remove §, digits, periods, and any trailing part after another § while preserving spaces
    name = re.sub(r'^§\s*\d+\.?\s*', '', name)
    name = re.sub(r'§.*$', '', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    # Remove all characters that are not English letters or spaces
    name = re.sub(r'[^a-zA-Z\s]', '', name)

    return name


# Function to calculate bigram frequencies
def calculate_bigrams(file_path):
    # Step 1: Read the Excel file
    data = pd.read_excel(file_path)

    # Step 2: Select relevant column
    hierarchy_data = data[['hierarchy_level_name']].copy()

    # Step 3: Apply cleaning function
    hierarchy_data['cleaned_hierarchy_level_name'] = hierarchy_data.apply(
        lambda row: clean_hierarchy_name_preserve_all_spaces(row), axis=1
    )

    # Step 4: Calculate bigram frequencies
    cleaned_texts = hierarchy_data['cleaned_hierarchy_level_name'].dropna().tolist()

    # Define stop words
    stop_words = set(ENGLISH_STOP_WORDS)

    # Calculate bigrams
    bigram_counts = Counter()
    for text in cleaned_texts:
        words = [word for word in text.split() if word.lower() not in stop_words]
        bigrams = zip(words, words[1:])
        bigram_counts.update(bigrams)

    # Convert to DataFrame
    bigram_df = pd.DataFrame(
        [(f"{w1} {w2}", count) for (w1, w2), count in bigram_counts.items()],
        columns=["Bigram", "Frequency"]
    )

    # Sort by frequency
    bigram_df = bigram_df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
    return bigram_df


# Calculate bigram frequencies for both files
bigram_2017 = calculate_bigrams(data_path_2017)
bigram_2018 = calculate_bigrams(data_path_2018)

# Combine results
combined_bigrams = pd.concat(
    [bigram_2017.rename(columns={"Bigram": "2017 Bigram", "Frequency": "2017 Frequency"}).reset_index(drop=True),
     bigram_2018.rename(columns={"Bigram": "2018 Bigram", "Frequency": "2018 Frequency"}).reset_index(drop=True)],
    axis=1
)

# Save combined bigram frequencies to an Excel file
output_file_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\N_GRAM\Titles_N_GRAM\Combined_Bigram_Frequencies.xlsx'
combined_bigrams.to_excel(output_file_path, index=False)
print(f"The combined bigram frequencies have been saved to: {output_file_path}")
