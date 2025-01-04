import pandas as pd
import re

# Load the bigram data
bigram_df = pd.read_excel("bigram_frequency_2017.xlsx")


# Define a cleaning function for bigrams
def clean_bigram(bigram):
    # Remove unwanted characters and fix spaces
    bigram = re.sub(r'[^a-zA-Z\s]', '', bigram)  # Keep only letters and spaces
    bigram = re.sub(r'\s{2,}', ' ', bigram)  # Remove extra spaces

    # Fix fragmented words (e.g., "Secre tary" -> "Secretary")
    bigram = re.sub(r'([a-zA-Z]+)\s([a-zA-Z]{1})\s([a-zA-Z]+)', r'\1\2\3', bigram)
    bigram = re.sub(r'([A-Z])\s([A-Z])', r'\1\2', bigram)  # Merge uppercase letters (e.g., "R ULES" -> "RULES")

    # Remove incomplete or meaningless bigrams
    words = bigram.split()
    if len(words) != 2:  # Ensure it's exactly two words
        return ""
    if any(len(word) < 2 for word in words):  # Remove bigrams with very short words
        return ""

    return bigram.strip()


# Apply cleaning function to the "Bigram" column
bigram_df['Cleaned Bigram'] = bigram_df['Bigram'].apply(clean_bigram)

# Remove rows where the cleaned bigram is empty
bigram_df = bigram_df[bigram_df['Cleaned Bigram'] != ""]

# Remove rows where the original "Bigram" column contains "Pub" or "title"
bigram_df = bigram_df[~bigram_df['Bigram'].str.contains(r'\b(Pub|title)\b', case=False, na=False)]

# Save the cleaned bigram data to a new Excel file
cleaned_bigram_file_path = "cleaned_bigram_frequency_2017_v2.xlsx"
bigram_df.to_excel(cleaned_bigram_file_path, index=False)

print(f"The cleaned bigram data has been saved to: {cleaned_bigram_file_path}")
