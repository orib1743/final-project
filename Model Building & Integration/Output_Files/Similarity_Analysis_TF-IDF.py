import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# טען את הנתונים
file_2017_path = r"C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017_fixed.xlsx"
file_2018_path = r"C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018_fixed.xlsx"

df_2017 = pd.read_excel(file_2017_path)
df_2018 = pd.read_excel(file_2018_path)

# נקה נתונים ריקים
df_2017_clean = df_2017.dropna(subset=['hierarchy_level_name', 'content'])
df_2018_clean = df_2018.dropna(subset=['hierarchy_level_name', 'content'])

# מציאת כותרות משותפות וייחודיות
titles_2017 = set(df_2017_clean['hierarchy_level_name'])
titles_2018 = set(df_2018_clean['hierarchy_level_name'])

common_titles = titles_2017.intersection(titles_2018)
unique_titles_2017 = titles_2017 - titles_2018
unique_titles_2018 = titles_2018 - titles_2017

print(f"כותרות משותפות: {len(common_titles)}")
print(f"כותרות ייחודיות ל-2017: {len(unique_titles_2017)}")
print(f"כותרות ייחודיות ל-2018: {len(unique_titles_2018)}")

# יצירת מילון של כותרות משותפות עם התוכן
content_2017 = {row['hierarchy_level_name']: row['content'] for _, row in df_2017_clean.iterrows() if row['hierarchy_level_name'] in common_titles}
content_2018 = {row['hierarchy_level_name']: row['content'] for _, row in df_2018_clean.iterrows() if row['hierarchy_level_name'] in common_titles}

# יצירת רשימות מקבילות
titles_common = list(content_2017.keys() & content_2018.keys())
texts_2017 = [content_2017[title] for title in titles_common]
texts_2018 = [content_2018[title] for title in titles_common]

# חישוב TF-IDF ודמיון קוסינוס
vectorizer = TfidfVectorizer()
tfidf_2017 = vectorizer.fit_transform(texts_2017)
tfidf_2018 = vectorizer.transform(texts_2018)

similarities = cosine_similarity(tfidf_2017, tfidf_2018).diagonal()

# יצירת DataFrame עם תוצאות ההשוואה
df_similarity = pd.DataFrame({
    "Title": titles_common,
    "Cosine Similarity": similarities
})

# הצגת סטטיסטיקות
print(df_similarity.describe())

# סינון סעיפים עם דמיון נמוך מ-0.9
df_low_similarity = df_similarity[df_similarity["Cosine Similarity"] < 0.9]

# הצגת סעיפים שהשתנו באופן משמעותי
print("\nסעיפים עם שינוי משמעותי:")
print(df_low_similarity)
