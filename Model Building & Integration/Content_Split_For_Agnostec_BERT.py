import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer

# טוען את ה-Tokenizer של BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# פונקציה לפיצול חכם של פסקאות
def split_text_into_chunks(text, max_length=512, overlap=100):
    """
    מפצל טקסט לקטעים של עד 512 טוקנים עם חפיפה של 100 טוקנים
    """
    if pd.isna(text) or len(text.strip()) == 0:
        return []

    # מחלק את הטקסט למשפטים
    sentences = sent_tokenize(text)

    # נבנה את הקטעים מחדש תוך שמירה על משפטים שלמים
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # מחשב את מספר הטוקנים במשפט
        sentence_tokens = tokenizer.tokenize(sentence)
        sentence_length = len(sentence_tokens)

        # אם הוספת המשפט חורגת מהגודל המקסימלי – נסיים את הקטע הנוכחי
        if current_length + sentence_length > max_length:
            chunks.append(current_chunk)
            # יוצרים קטע חדש שמתחיל עם חפיפה של `overlap` טוקנים
            overlap_tokens = tokenizer.tokenize(" ".join(current_chunk[-overlap:]))
            current_chunk = overlap_tokens
            current_length = len(overlap_tokens)

        # מוסיפים את המשפט לקטע הנוכחי
        current_chunk.append(sentence)
        current_length += sentence_length

    # מוסיפים את הקטע האחרון
    if current_chunk:
        chunks.append(current_chunk)

    # ממירים לרשימה של טקסטים (ולא רשימה של רשימות)
    return [" ".join(chunk) for chunk in chunks]

# טוען את קובצי האקסל
file_2017_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017_fixed.xlsx'
file_2018_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018_fixed.xlsx'

# קריאת הנתונים
data_2017 = pd.read_excel(file_2017_path)
data_2018 = pd.read_excel(file_2018_path)

# מחלקים את הפסקאות הגדולות לקטעים קטנים עם חפיפה
data_2017["split_content"] = data_2017["content"].apply(split_text_into_chunks)
data_2018["split_content"] = data_2018["content"].apply(split_text_into_chunks)

# הופכים רשימת קטעי טקסט למחרוזת מופרדת ב-"\n\n"
data_2017["split_content"] = data_2017["split_content"].apply(lambda x: "\n\n".join(x))
data_2018["split_content"] = data_2018["split_content"].apply(lambda x: "\n\n".join(x))

# שמירת הנתונים המחולקים לקובץ אקסל
output_file_2017 = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Output_Files\Extracted_Hierarchy_Content_2017_Split.xlsx'
output_file_2018 = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Output_Files\Extracted_Hierarchy_Content_2018_Split.xlsx'


data_2017.to_excel(output_file_2017, index=False)
data_2018.to_excel(output_file_2018, index=False)

print(f" קובץ שמור: {output_file_2017}")
print(f" קובץ שמור: {output_file_2018}")
