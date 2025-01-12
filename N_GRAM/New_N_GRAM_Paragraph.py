import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

# File paths for 2017 and 2018 data
file_path_2017 = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017.xlsx'
file_path_2018 = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018.xlsx'

# Function to clean content
def clean_text(text):
    # Merge hyphenated words and line breaks
    text = re.sub(r'-\s*\n*\s*', '', text)
    # Remove parentheses and their content
    text = re.sub(r'\(.*?\)', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove Roman numerals (if standalone)
    text = re.sub(r'\b(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)\b', '', text)
    # Merge split letters into words
    text = re.sub(r'\b(\w)\s+(\w)\b', r'\1\2', text)
    # Remove general stop words
    text = re.sub(r'\b(paragraph|subparagraph|section|part|subpart|chapter|subchapter|subtitle|title|on|it|in|the|and|of|to|with|by|as|for|or|if|is|at)\b', '', text, flags=re.IGNORECASE)
    # Remove irrelevant terms (e.g., Page, TITLE)
    text = re.sub(r'\b(Page|TITL|INTERNAL|REVENUE|CODE)\b', '', text, flags=re.IGNORECASE)
    return text

# Function to calculate trigrams
def calculate_trigrams(file_path):
    # Load the content
    data = pd.read_excel(file_path)
    data_selected = data[['content']].dropna()

    # Clean the content
    data_selected['cleaned_content'] = data_selected['content'].apply(clean_text)

    # Create trigrams using CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')
    ngrams_matrix = vectorizer.fit_transform(data_selected['cleaned_content'])
    ngrams = vectorizer.get_feature_names_out()
    frequencies = ngrams_matrix.sum(axis=0).A1

    # Convert to DataFrame and sort
    ngrams_df = pd.DataFrame({'n-gram': ngrams, 'frequency': frequencies}).sort_values(by='frequency', ascending=False)

    # Filter rows containing "pub"
    ngrams_df = ngrams_df[~ngrams_df['n-gram'].str.contains(r'\bpub\b', case=False)]
    return ngrams_df

# Process 2017 and 2018 files
trigram_2017 = calculate_trigrams(file_path_2017)
trigram_2018 = calculate_trigrams(file_path_2018)

# Combine results
combined_trigrams = pd.concat(
    [trigram_2017.rename(columns={"n-gram": "2017 Trigram", "frequency": "2017 Frequency"}).reset_index(drop=True),
     trigram_2018.rename(columns={"n-gram": "2018 Trigram", "frequency": "2018 Frequency"}).reset_index(drop=True)],
    axis=1
)

# Sort consistently by both 2017 and 2018 frequencies in descending order
combined_trigrams = combined_trigrams.sort_values(by=["2017 Frequency", "2018 Frequency"], ascending=[False, False])

# Save combined trigrams to an Excel file
output_file_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\N_GRAM\Combined_Trigram_Frequencies.xlsx'
combined_trigrams.to_excel(output_file_path, index=False)
print(f"The updated combined trigram frequencies have been saved to: {output_file_path}")
