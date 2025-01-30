import re
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from difflib import SequenceMatcher

# 🔹 טוען את המודל וה-Tokenizer של BERT
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# שימוש ב-GPU אם קיים
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔹 טוען את קובצי האקסל עם הפסקאות המחולקות
file_2017_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Output_Files\Extracted_Hierarchy_Content_2017_Split.xlsx'
file_2018_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Output_Files\Extracted_Hierarchy_Content_2018_Split.xlsx'

# קריאת הנתונים
data_2017 = pd.read_excel(file_2017_path)
data_2018 = pd.read_excel(file_2018_path)

# ✅ פונקציה לזיהוי שינויי מספור בלבד
def detect_numbering_change(text_2017, text_2018):
    """
    מזהה אם השינוי הוא רק שינוי מספר סעיף, אך לא שינוי תוכן.
    """
    if pd.isna(text_2017) or pd.isna(text_2018):
        return False

    # מזהה סימוני סעיפים כמו §28, §45C, §30A וכו'
    num_pattern = r'§\d+[A-Z]*'
    numbers_2017 = re.findall(num_pattern, text_2017)
    numbers_2018 = re.findall(num_pattern, text_2018)

    # אם רק מספרי הסעיפים השתנו והתוכן דומה ב-90%+, נחשיב את זה כרה-ארגון בלבד
    if numbers_2017 != numbers_2018:
        similarity = SequenceMatcher(None, text_2017, text_2018).ratio()
        return similarity > 0.9

    return False

# 🔹 פונקציה לחישוב אמבדינגים של BERT
def compute_bert_embedding(text):
    """
    מקבלת טקסט ומחזירה וקטור של 768 מספרים מתוך BERT
    """
    if pd.isna(text) or len(text.strip()) == 0:
        return np.zeros(768)

    # טוקניזציה
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # העברת הקלט דרך המודל
    with torch.no_grad():
        outputs = model(**inputs)

    # מחזירים את ה-CLS embedding
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

# 🔹 חישוב האמבדינגים לכל הפסקאות בעמודת `split_content`
data_2017["bert_embeddings"] = data_2017["split_content"].apply(compute_bert_embedding)
data_2018["bert_embeddings"] = data_2018["split_content"].apply(compute_bert_embedding)

# 🔹 השוואת טקסטים על בסיס דמיון קוסינוסי
similarity_scores = []
numbering_changes = []
for i, row_2017 in data_2017.iterrows():
    if i < len(data_2018):  # נוודא שהאינדקסים תואמים
        embedding_2017 = row_2017["bert_embeddings"].reshape(1, -1)
        embedding_2018 = data_2018.loc[i, "bert_embeddings"].reshape(1, -1)

        similarity = cosine_similarity(embedding_2017, embedding_2018)[0][0]
        similarity_scores.append(similarity)

        # בדיקה אם מדובר רק בשינויי מספור
        numbering_changes.append(detect_numbering_change(row_2017["split_content"], data_2018.loc[i, "split_content"]))
    else:
        similarity_scores.append(np.nan)
        numbering_changes.append(False)

# 🔹 הוספת עמודות דמיון וזיהוי שינויי מספור
data_2017["similarity_2018"] = similarity_scores
data_2017["numbering_change"] = numbering_changes

# 🔹 מציאת פסקאות עם שינוי משמעותי (ללא שינוי מספור בלבד)
data_2017["change_detected"] = (data_2017["similarity_2018"] < 0.7) & (~data_2017["numbering_change"])

# ✅ שמירת ההשוואה
output_comparison_file = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Output_Files\Content_BERT_Comparison.xlsx'
data_2017.to_excel(output_comparison_file, index=False)


print(f"✅ קובץ עם השוואת השינויים נשמר: {output_comparison_file}")
