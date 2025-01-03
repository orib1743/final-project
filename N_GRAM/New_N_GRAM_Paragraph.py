import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

# שלב 1: טעינת הקובץ
file_path = 'Extracted_Hierarchy_Data_With_Word_Count_Paragraph_2018.xlsx'  # שנה את הנתיב לקובץ שלך
data = pd.read_excel(file_path)

# שלב 2: סינון שורות שבהן `hierarchy_level` הוא `7.Paragraph`
data_filtered = data[data['hierarchy_level'] == '7.Paragraph']

# שלב 3: שמירת עמודות `hierarchy_level_name` ו-`content` בלבד
data_selected = data_filtered[['hierarchy_level_name', 'content']].dropna()

# שלב 4: איחוד העמודות
data_selected['combined'] = data_selected['hierarchy_level_name'] + " " + data_selected['content']

# שלב 5: ניקוי הטקסט
def clean_text(text):
    text = re.sub(r'^\(\w+\)\s*', '', text)  # הסרת סוגריים בתחילת הפסקה
    text = re.sub(r'[^\w\s]', '', text)  # הסרת סימני פיסוק
    text = re.sub(r'\d+', '', text)  # הסרת מספרים
    text = re.sub(r'\b(paragraph|subparagraph|section|part|subpart|chapter|subchapter|subtitle)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()  # הסרת רווחים מיותרים
    return text

data_selected['cleaned'] = data_selected['combined'].apply(clean_text)

# שלב 6: יצירת n-gram עם n=3
vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')
ngrams_matrix = vectorizer.fit_transform(data_selected['cleaned'])
ngrams = vectorizer.get_feature_names_out()
frequencies = ngrams_matrix.sum(axis=0).A1

# שלב 7: שמירת התוצאות לקובץ Excel
ngrams_df = pd.DataFrame({'n-gram': ngrams, 'frequency': frequencies}).sort_values(by='frequency', ascending=False)
output_path = 'new_Paragraph_ngrams_frequencies_2018.xlsx'  # נתיב לשמירת הקובץ
ngrams_df.to_excel(output_path, index=False)

print(f"File saved at {output_path}")
