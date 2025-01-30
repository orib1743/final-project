import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from rapidfuzz import fuzz

# טען את הנתונים
file_2017_path = r"C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017_fixed.xlsx"
file_2018_path = r"C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018_fixed.xlsx"

df_2017 = pd.read_excel(file_2017_path)
df_2018 = pd.read_excel(file_2018_path)

# נקה נתונים ריקים
df_2017_clean = df_2017.dropna(subset=['hierarchy_level_name', 'content'])
df_2018_clean = df_2018.dropna(subset=['hierarchy_level_name', 'content'])

# מציאת התאמות בין כותרות באמצעות fuzzywuzzy ו-Agnostic BERT
titles_2017 = list(df_2017_clean['hierarchy_level_name'])
titles_2018 = list(df_2018_clean['hierarchy_level_name'])

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def find_best_match(title, candidates, threshold=80):
    """ מחפש את ההתאמה הטובה ביותר בעזרת fuzzywuzzy ובמידת הצורך BERT """
    best_match = None
    best_score = 0
    best_bert_score = 0

    for candidate in candidates:
        fw_score = fuzz.token_sort_ratio(title, candidate)  # השוואה מבוססת מילים

        if fw_score > best_score:
            best_score = fw_score
            best_match = candidate

            # אם fuzzywuzzy מצא התאמה מעל סף מסוים, נחשב גם דמיון באמצעות BERT
            if best_score >= threshold:
                embeddings = model.encode([title, candidate], convert_to_tensor=True)
                bert_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

                if bert_score > best_bert_score:
                    best_bert_score = bert_score

    return best_match, best_score, best_bert_score


# מציאת כותרות תואמות בין 2017 ל-2018
matched_titles = {}

title_pairs = []
fuzzy_scores = []
bert_scores = []

for title in titles_2017:
    best_match, fw_score, bert_score = find_best_match(title, titles_2018)
    if best_match:
        matched_titles[title] = best_match
        title_pairs.append((title, best_match))
        fuzzy_scores.append(fw_score)
        bert_scores.append(bert_score)

# יצירת DataFrame עם ההתאמות
matched_df = pd.DataFrame(title_pairs, columns=["Title 2017", "Title 2018"])
matched_df["Fuzzy Score"] = fuzzy_scores
matched_df["BERT Similarity"] = bert_scores

# שמירת הכותרות המשויכות לקובץ אקסל
matched_df.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\Matched_Titles_Fuzzy.xlsx", index=False)

print("התאמות בין כותרות נשמרו בקובץ Matched_Titles.xlsx")

# עדכון סטים של כותרות משותפות וייחודיות
common_titles = set(matched_df["Title 2017"])  # כותרות משותפות שזוהו
unique_titles_2017 = set(titles_2017) - common_titles
unique_titles_2018 = set(titles_2018) - set(matched_df["Title 2018"])

print(f"כותרות משותפות שנמצאו: {len(common_titles)}")
print(f"כותרות ייחודיות ל-2017: {len(unique_titles_2017)}")
print(f"כותרות ייחודיות ל-2018: {len(unique_titles_2018)}")

# יצירת DataFrame של כותרות ייחודיות ושמירתן לקובץ Excel
df_unique_2017 = df_2017_clean[df_2017_clean['hierarchy_level_name'].isin(unique_titles_2017)]
df_unique_2018 = df_2018_clean[df_2018_clean['hierarchy_level_name'].isin(unique_titles_2018)]
df_common = df_2017_clean[df_2017_clean['hierarchy_level_name'].isin(common_titles)]

df_unique_2017.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\Unique_Titles_2017_Fuzzy.xlsx", index=False)
df_unique_2018.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\Unique_Titles_2018_Fuzzy.xlsx", index=False)
df_common.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\Common_Titles_Fuzzy.xlsx", index=False)

print("כותרות ייחודיות ומשותפות נשמרו לקובצי אקסל")

# פונקציה לחלוקת טקסט ארוך לפסקאות קטנות
def split_text(text, max_length=256):
    sentences = text.split(". ")  # חלוקה למשפטים
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# יצירת מילון של כותרות משותפות עם תוכן
content_2017 = {row['hierarchy_level_name']: split_text(row['content']) for _, row in df_2017_clean.iterrows() if
                row['hierarchy_level_name'] in matched_titles.keys()}
content_2018 = {row['hierarchy_level_name']: split_text(row['content']) for _, row in df_2018_clean.iterrows() if
                row['hierarchy_level_name'] in matched_titles.values()}


def compute_similarity(chunks_2017, chunks_2018):
    """ מחשב דמיון בין רשימות של קטעי טקסט, מחזיר גם את הפסקאות השונות ביותר """
    embeddings_2017 = model.encode(chunks_2017, convert_to_tensor=True)
    embeddings_2018 = model.encode(chunks_2018, convert_to_tensor=True)

    similarity_matrix = util.pytorch_cos_sim(embeddings_2017, embeddings_2018).cpu().numpy()
    min_index = np.unravel_index(np.argmin(similarity_matrix, axis=None), similarity_matrix.shape)

    return np.mean(similarity_matrix), similarity_matrix, chunks_2017[min_index[0]], chunks_2018[min_index[1]]


# חישוב הדמיון לכל סעיף
similarities = []
titles_common = []
detail_changes = []
changed_text_2017 = []
changed_text_2018 = []

for title in matched_titles.keys():
    sim, sim_matrix, text_2017, text_2018 = compute_similarity(content_2017[title], content_2018[matched_titles[title]])
    similarities.append(sim)
    titles_common.append(title)

    # מציאת הפסקאות שהכי השתנו
    min_sim = np.min(sim_matrix)
    detail_changes.append(f"Min Similarity: {min_sim:.3f}")
    changed_text_2017.append(text_2017)
    changed_text_2018.append(text_2018)

# יצירת DataFrame עם תוצאות ההשוואה
df_similarity = pd.DataFrame({
    "Title": titles_common,
    "Cosine Similarity": similarities,
    "Change Details": detail_changes,
    "Text 2017": changed_text_2017,
    "Text 2018": changed_text_2018
})

# הצגת סטטיסטיקות
print(df_similarity.describe())

# סינון סעיפים עם דמיון נמוך מ-0.9
df_low_similarity = df_similarity[df_similarity["Cosine Similarity"] < 0.9]

# שמירת התוצאות לקובץ אקסל
df_similarity.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\Similarity_Results_Fuzzy.xlsx", index=False)

print("תוצאות השוואת הטקסטים נשמרו בקובץ Similarity_Results.xlsx")

# הצגת סעיפים שהשתנו באופן משמעותי
print("\nסעיפים עם שינוי משמעותי:")
print(df_low_similarity)