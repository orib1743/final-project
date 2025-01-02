import re
from PyPDF2 import PdfReader
import pandas as pd


def extract_complete_hierarchy(file_path):
    # Read the PDF, skipping the first 22 pages
    reader = PdfReader(file_path)
    content_with_pages = []

    for page_num, page in enumerate(reader.pages):
        if page_num > 21:  # Skip the first 22 pages
            text = page.extract_text()
            content_with_pages.append((page_num + 1, text))  # Store page number (1-based) and text

    # Define patterns for different hierarchy levels
    patterns = {
        "1.Subtitle": r"Subtitle\s+[A-Z]+\—.*",  # Matches "Subtitle A—Income Taxes"
        "2.Chapter": r"CHAPTER\s+\d+[A-Z]?—.*",  # Matches "CHAPTER 2—NAME" and "CHAPTER 2A—NAME"
        "3.Subchapter": r"Subchapter\s+[A-Z]+\—.*",  # Matches "Subchapter A—NAME"
        "4.Part": r"(?<!SUB)PART\s+[A-Z]+\—.*",  # Matches "PART A—NAME" but not "SUBPART"
        "5.Subpart": r"SUBPART\s+[A-Z]+\—.*",  # Matches "SUBPART A—NAME"
        "6.Section": r"§\s*\d+[A-Z]?\. .*?(?=\n|$)",  # Matches "§25A. " and "§25. "
        "7.Paragraph": r"\([a-z]+\) .*",  # Matches "(a) Title" or "(1) Title"
    }

    # List of reserved words that indicate the start or end of a title
    reserved_words = ["CHAPTER", "Chapter", "Subchapter", "PART", "Sec.", "SUBPART", "Section", "Part", "Subpart",
                      "(a)"]

    # Store hierarchy data
    hierarchy_data = []
    last_hierarchy = {}

    # Process content for each page
    for page_num, page_content in content_with_pages:
        lines = page_content.splitlines()
        processed_content = []
        skip_next_line = False

        for i, line in enumerate(lines):
            if skip_next_line:
                skip_next_line = False
                continue

            # Check if line is part of a multi-line title
            if any(re.match(pattern, line) for pattern in patterns.values()):
                # Check if the next line continues the title
                if i + 1 < len(lines) and not any(re.match(pattern, lines[i + 1]) for pattern in patterns.values()):
                    line = line + " " + lines[i + 1].strip()
                    skip_next_line = True

                # Remove reserved words or numbers starting from the third word of the title
                words = line.strip().split()  # Split the line into words
                for idx, word in enumerate(words):
                    # Check for reserved words starting from the third word
                    if idx >= 2 and word in reserved_words:
                        line = " ".join(words[:idx]).strip()  # Keep only the words up to the reserved word
                        break

                    # Check for numbers starting from the third word
                    if idx >= 2 and re.match(r'^\d+\..*$', word):  # Match a whole number
                        line = " ".join(words[:idx]).strip()  # Keep only the words up to the number
                        break

            processed_content.append(line)

        processed_content = "\n".join(processed_content)

        # Match hierarchy levels
        for level, pattern in patterns.items():
            matches = list(re.finditer(pattern, processed_content))
            for i, match in enumerate(matches):
                start_pos, end_pos = match.start(), match.end()
                title_or_content = match.group().strip()

                # Determine parent hierarchy level and name
                previous_level = None
                previous_name = None
                for prev_level, prev_data in reversed(last_hierarchy.items()):
                    if prev_level < level:  # Find the correct parent level
                        previous_level = prev_level
                        previous_name = prev_data["name"]
                        break

                # Update the last hierarchy tracker
                last_hierarchy[level] = {"type": level, "name": title_or_content}

                # Calculate the number of words in the current title
                current_title_word_count = len(title_or_content.split())

                # Calculate the content and words up to the next hierarchy level
                if i + 1 < len(matches):
                    next_start_pos = matches[i + 1].start()
                else:
                    # If this is the last match, search for the next hierarchy in the rest of the file
                    next_start_pos = len(processed_content)
                    for next_level, next_pattern in patterns.items():
                        if next_level != level:  # Skip the same level
                            next_match = re.search(next_pattern, processed_content[end_pos:])
                            if next_match:
                                next_start_pos = min(next_start_pos, end_pos + next_match.start())

                content_between = processed_content[end_pos:next_start_pos].strip()
                word_count = len(content_between.split())

                # Append the structured data
                hierarchy_data.append({
                    "hierarchy_level": level,
                    "hierarchy_level_name": title_or_content,
                    "last_hierarchy_level": previous_level,
                    "last_hierarchy_level_name": previous_name,
                    "title_type": level,
                    "current_title_word_count": current_title_word_count,  # Number of words in the title
                    "word_count_to_next": word_count,  # Number of words to the next title
                    "content": content_between,  # Extracted content up to the next hierarchy
                    "page_number": page_num  # Store the page number
                })

    # Convert to DataFrame for further processing
    df = pd.DataFrame(hierarchy_data)
    return df


# Path to the PDF document
pdf_path = "D:\\yifat\\Extract_Data_Project\\USCODE-2017-title26.pdf"

# Extract the structured hierarchy
extracted_hierarchy = extract_complete_hierarchy(pdf_path)

# Save the extracted data to an Excel file
output_file = "D:\\yifat\\Extract_Data_Project\\Extracted_Hierarchy_Data_With_Word_Count_Paragraph_2017.xlsx"
extracted_hierarchy.to_excel(output_file, index=False)

output_file
