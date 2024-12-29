import re
from collections import Counter
from itertools import islice
import pandas as pd

# Function to clean text
def clean_text(file_path):
    """
    Cleans the input text by removing special characters, digits, extra spaces, and common stopwords.
    Args:
        file_path (str): Path to the input text file.
    Returns:
        str: Cleaned text.
    """
    stopwords = set([
        "the", "is", "in", "and", "to", "a", "of", "on", "with", "this", "that", "for", "it", "as", "are", "an", "at", "by", "from", "or", "be", "was", "were", "has", "have", "not", "but", "will", "we", "can","section","title","sections"
    ])  # Add Hebrew or other relevant stopwords as needed

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Remove non-alphanumeric characters except Hebrew letters and spaces
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove stopwords
        text = " ".join(word for word in text.split() if word.lower() not in stopwords)

        return text
    except Exception as e:
        print(f"Error: {e}")
        return ""

# Function to generate n-grams
def generate_ngrams(text, n):
    """
    Generates n-grams from the input text.
    Args:
        text (str): Input cleaned text.
        n (int): Size of the n-grams.
    Returns:
        Counter: Frequency count of n-grams.
    """
    words = text.split()
    ngrams = zip(*[islice(words, i, None) for i in range(n)])
    return Counter([" ".join(gram) for gram in ngrams])

# Main function to clean text, generate n-grams, and save to Excel
def main(file_path, n, output_file):
    """
    Main function to clean text, compute n-grams, and save to Excel.
    Args:
        file_path (str): Path to the input text file.
        n (int): Size of the n-grams.
        output_file (str): Path to the output Excel file.
    Returns:
        None
    """
    # Step 1: Clean the text
    cleaned_text = clean_text(file_path)

    # Step 2: Generate n-grams
    ngrams = generate_ngrams(cleaned_text, n)

    # Step 3: Save n-grams and their frequencies to Excel
    ngrams_df = pd.DataFrame(ngrams.items(), columns=["n-gram", "Frequency"])
    ngrams_df.sort_values(by="Frequency", ascending=False, inplace=True)
    ngrams_df.to_excel(output_file, index=False)
    print(f"n-grams and their frequencies have been saved to {output_file}")

if __name__ == "__main__":
    # Specify the file path, n for n-grams, and output file path
    file_path = "Paragraphs 2018.txt"  # Replace with your file path
    n = 3  # Change n to the desired value for n-grams
    output_file = "ngrams_paragraph_frequencies_2018.xlsx"  # Replace with your desired output file path
    main(file_path, n, output_file)
