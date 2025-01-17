import pandas as pd
from docx import Document

# Load the Excel file containing the hierarchy levels and titles
excel_file = "D:\\yifat\\Final_Project\\Data Collection & Preprocessing\\Headlines 2017.xlsx"
df = pd.read_excel(excel_file)

# Extract the relevant columns from the Excel file
titles = df['hierarchy_level_name'].tolist()

# Load the Word document
word_file = "D:\\yifat\\Final_Project\\Data Collection & Preprocessing\\TITLE 26_2017_new2.docx"
doc = Document(word_file)

# Define the strings that indicate stopping points
stop_strings = ["(Aug. 16, 1954", "(Added Pub. L."]

# Initialize variables to store extracted data
content_list = []
title_index = 0  # Start with the first title in the list
start_paragraph_index = 0  # Track where to start searching in the Word document

# Function to save the current content when switching to a new hierarchy
def save_content(hierarchy, content, content_list):
    if hierarchy and content:
        content_text = "\n".join(content).strip()
        content_list.append({
            'hierarchy_level_name': hierarchy,
            'content': content_text,
            'word_count': len(content_text.split())
        })

# Iterate through the paragraphs in the Word document
paragraphs = doc.paragraphs  # Store all paragraphs for easier access
num_paragraphs = len(paragraphs)

while title_index < len(titles):
    current_hierarchy = titles[title_index]
    current_content = []
    found_title = False

    # Iterate through the paragraphs starting from the last position
    for i in range(start_paragraph_index, num_paragraphs):
        para_text = paragraphs[i].text.strip()

        # Skip empty paragraphs
        if not para_text:
            continue

        # If the paragraph matches the current title
        if para_text == current_hierarchy and not found_title:
            found_title = True
            start_paragraph_index = i + 1  # Update the start position
            continue

        # Collect content after the title is found
        if found_title:
            # Stop if we reach the next title or a stop string
            if ((title_index + 1 < len(titles)) and para_text == titles[title_index + 1]):
                start_paragraph_index = i  # Update the start position
                break
            if any(stop_string in para_text for stop_string in stop_strings):
                start_paragraph_index = i + 1  # Update the start position
                break

            # Append the content
            current_content.append(para_text)

    # Move to the next title in the list
    title_index += 1  # Advance to the next title in Excel
    found_title = False
    # Save the content for the current title
    save_content(current_hierarchy, current_content, content_list)
    # Move to the next title in the list
    #title_index += 1  # Advance to the next title in Excel

# Convert the extracted data to a DataFrame
output_df = pd.DataFrame(content_list)

# Add the original Excel columns to the DataFrame
output_df = df.merge(output_df, on='hierarchy_level_name', how='left')

# Save the output to a new Excel file
output_file = "D:\\yifat\\Final_Project\\Data Collection & Preprocessing\\Extracted_Hierarchy_Content_2017_fixed_without_duplicates.xlsx"
output_df.to_excel(output_file, index=False)

print(f"Extracted content saved to {output_file}")
