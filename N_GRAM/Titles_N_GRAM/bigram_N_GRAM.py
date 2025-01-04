import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Step 1: Read the Excel fileA
data = pd.read_excel("Extracted_Hierarchy_Data_With_Ending_Check_2017.xlsx")

# Step 2: Select relevant columns and create a DataFrame
hierarchy_data = data[['hierarchy_level', 'hierarchy_level_name']].copy()


# Step 3: Define the cleaning function to strictly preserve original spacing
def clean_hierarchy_name_preserve_all_spaces(row):
    level = row['hierarchy_level']
    name = row['hierarchy_level_name']

    # Standardize dashes
    name = re.sub(r'[–—]', '-', name)

    # Normalize the hierarchy level to lowercase for consistent checks
    level_lower = level.lower()

    # Clean based on hierarchy type
    if any(x in level_lower for x in ['subtitle', 'chapter', 'subchapter', 'part', 'subpart']):
        # Remove prefixes but retain original spacing
        name = re.sub(r'^(Subtitle|Chapter|Subchapter|Part|Subpart) [A-Z\d]+\s*[-–—]\s*', '', name, flags=re.IGNORECASE)
    elif 'section' in level_lower:
        # Remove §, digits, periods, and any trailing part after another § while preserving spaces
        name = re.sub(r'^§\s*\d+\.?\s*', '', name)
        name = re.sub(r'§.*$', '', name)
        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name).strip()
        # Remove all characters that are not English letters or spaces
        name = re.sub(r'[^a-zA-Z\s]', '', name)
    # Return the name with all original spacing preserved
    return name


# Step 4: Apply the cleaning function
hierarchy_data['cleaned_hierarchy_level_name'] = hierarchy_data.apply(clean_hierarchy_name_preserve_all_spaces, axis=1)

# Save the cleaned data to a new Excel file
cleaned_file_path = "cleaned_hierarchy_data_2017.xlsx"

# Step 5: Calculate bigram frequencies directly from cleaned_hierarchy_level_name
cleaned_texts = hierarchy_data['cleaned_hierarchy_level_name'].dropna().tolist()

# Step 6: Define a stop words list
stop_words = set(ENGLISH_STOP_WORDS)

# Step 7: Filter out stop words before calculating bigrams
bigram_counts_filtered = Counter()
for text in cleaned_texts:
    # Split the text into words and filter out stop words
    words = [word for word in text.split() if word.lower() not in stop_words]
    bigrams = zip(words, words[1:])  # Create bigrams
    bigram_counts_filtered.update(bigrams)

# Convert filtered bigram counts to a DataFrame
bigram_df_filtered = pd.DataFrame(
    [(f"{w1} {w2}", count) for (w1, w2), count in bigram_counts_filtered.items()],
    columns=["Bigram", "Frequency"]
)

# Sort by frequency in descending order
bigram_df_filtered = bigram_df_filtered.sort_values(by="Frequency", ascending=False).reset_index(drop=True)

# Save the filtered bigram frequencies to a new Excel file
bigram_filtered_file_path = "bigram_frequency_2017.xlsx"
bigram_df_filtered.to_excel(bigram_filtered_file_path, index=False)

print(f"The filtered bigram frequencies have been saved to: {bigram_filtered_file_path}")
