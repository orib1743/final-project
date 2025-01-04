import pandas as pd
import re

# Load the n-gram data
ngram_df = pd.read_excel("new_Paragraph_ngrams_frequencies_2017.xlsx")

# Define a cleaning function for n-grams
def clean_ngram(ngram):
    # Remove unwanted characters and fix spaces
    ngram = re.sub(r'[^a-zA-Z\s]', '', ngram)  # Keep only letters and spaces
    ngram = re.sub(r'\s{2,}', ' ', ngram)  # Remove extra spaces

    # Fix fragmented words (e.g., "Secre tary" -> "Secretary")
    ngram = re.sub(r'([a-zA-Z]+)\s([a-zA-Z]{1})\s([a-zA-Z]+)', r'\1\2\3', ngram)
    ngram = re.sub(r'([A-Z])\s([A-Z])', r'\1\2', ngram)  # Merge uppercase letters (e.g., "R ULES" -> "RULES")

    # Remove incomplete or meaningless n-grams
    words = ngram.split()
    if len(words) != 3:  # Ensure it's exactly three words
        return ""
    if any(len(word) < 2 for word in words):  # Remove n-grams with very short words
        return ""

    return ngram.strip()

# Apply the cleaning function to the "n-gram" column
ngram_df['Cleaned n-gram'] = ngram_df['n-gram'].apply(clean_ngram)

# Remove rows where the cleaned n-gram is empty
ngram_df = ngram_df[ngram_df['Cleaned n-gram'] != ""]

# Remove rows where the original "n-gram" column contains "Pub" or "title"
ngram_df = ngram_df[~ngram_df['n-gram'].str.contains(r'\bpub\b', case=False, na=False)]
ngram_df = ngram_df[~ngram_df['n-gram'].str.contains(r'\btitle\b', case=False, na=False)]

# Save the cleaned n-gram data to a new Excel file
cleaned_ngram_file_path = "cleaned_Paragraph_ngrams_frequencies_2017_v2.xlsx"
ngram_df.to_excel(cleaned_ngram_file_path, index=False)

print(f"The cleaned n-gram data has been saved to: {cleaned_ngram_file_path}")
