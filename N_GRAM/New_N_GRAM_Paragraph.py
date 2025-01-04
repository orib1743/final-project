import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

# שלב 1: טעינת הקובץ
file_path = 'Extracted_Hierarchy_Data_With_Word_Count_Paragraph_2017.xlsx'  # שנה את הנתיב לקובץ שלך
data = pd.read_excel(file_path)

# שלב 2: סינון שורות שבהן `hierarchy_level` הוא `7.Paragraph`
data_filtered = data[data['hierarchy_level'] == '7.Paragraph']

# שלב 3: שמירת עמודות `hierarchy_level_name` ו-`content` בלבד
data_selected = data_filtered[['hierarchy_level_name', 'content']].dropna()

# שלב 4: איחוד העמודות
data_selected['combined'] = data_selected['hierarchy_level_name'] + " " + data_selected['content']

# שלב 5: ניקוי הטקסט
def clean_text(text):
    # איחוד מילים שמופרדות במקף ושבירת שורה
    text = re.sub(r'-\s*\n*\s*', '', text)
    # הסרת סוגריים ותוכן בתוכם
    text = re.sub(r'\(.*?\)', '', text)
    # הסרת מספרים
    text = re.sub(r'\d+', '', text)
    # הסרת סימני פיסוק מיותרים ותווים מיוחדים
    text = re.sub(r'[^\w\s]', '', text)
    # איחוד רווחים בתוך מילים
    text = re.sub(r'\b(\w)\s+(\w)\b', r'\1\2', text)
    # הסרת מילים כלליות
    text = re.sub(r'\b(paragraph|subparagraph|section|part|subpart|chapter|subchapter|subtitle|title)\b', '', text, flags=re.IGNORECASE)
    # הסרת אזכורים לא רלוונטיים (כמו Page TITL וכו')
    text = re.sub(r'\b(Page|TITL|INTERNAL|REVENUE|CODE)\b', '', text, flags=re.IGNORECASE)
    # הסרת רווחים מיותרים
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data_selected['cleaned_content'] = data_selected['combined'].apply(clean_text)

# שלב 6: שמירת הנתונים לאחר הניקוי
cleaned_file_path = 'Cleaned_Paragraph_Hierarchy_Data_2017.xlsx'  # נתיב לשמירת הקובץ
data_selected.to_excel(cleaned_file_path, index=False)
print(f"File with cleaned content saved at {cleaned_file_path}")

# שלב 7: יצירת n-gram עם n=3 על העמודה הנקייה
vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')
ngrams_matrix = vectorizer.fit_transform(data_selected['cleaned_content'])
ngrams = vectorizer.get_feature_names_out()
frequencies = ngrams_matrix.sum(axis=0).A1

# שלב 8: שמירת תוצאות ה-n-grams לקובץ Excel
ngrams_df = pd.DataFrame({'n-gram': ngrams, 'frequency': frequencies}).sort_values(by='frequency', ascending=False)
ngrams_output_path = 'new_Paragraph_ngrams_frequencies_2017.xlsx'  # נתיב לשמירת הקובץ
ngrams_df.to_excel(ngrams_output_path, index=False)
print(f"File with n-gram frequencies saved at {ngrams_output_path}")
