import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import util
from nltk.util import ngrams
from difflib import SequenceMatcher

# טען את הנתונים
file_2017_path = r"C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017_fixed.xlsx"
file_2018_path = r"C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018_fixed.xlsx"

df_2017 = pd.read_excel(file_2017_path)
df_2018 = pd.read_excel(file_2018_path)

# נקה נתונים ריקים
df_2017_clean = df_2017.dropna(subset=['hierarchy_level_name', 'content'])
df_2018_clean = df_2018.dropna(subset=['hierarchy_level_name', 'content'])

# מציאת התאמות בין כותרות באמצעות fuzzywuzzy ו-LaBSE
titles_2017 = list(df_2017_clean['hierarchy_level_name'])
titles_2018 = list(df_2018_clean['hierarchy_level_name'])

# טען את מודל LaBSE
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE", do_lower_case=False)
model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def encode_text(texts):
    """ מקודד רשימה של טקסטים בעזרת LaBSE """
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # שימוש ב-last_hidden_state במקום pooler_output
    return embeddings


def jaccard_similarity(str1, str2):
    """ מחשב דמיון Jaccard בין שתי מחרוזות לפי קבוצות מילים """
    set1, set2 = set(str1.split()), set(str2.split())
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

def levenshtein_distance(str1, str2):
    """ מחשב את מרחק Levenshtein בין שתי מחרוזות """
    return len(str1) + len(str2) - 2 * SequenceMatcher(None, str1, str2).ratio() * min(len(str1), len(str2))

def find_best_match(title, candidates, fuzzy_threshold=80, jaccard_threshold=0.7, levenshtein_threshold=5):
    """ מחפש את ההתאמה הטובה ביותר בעזרת Fuzzy, Jaccard, Levenshtein ובמידת הצורך LaBSE """
    best_match = None
    best_fuzzy_score = 0
    best_bert_score = 0
    best_jaccard = 0
    best_levenshtein = float('inf')

    for candidate in candidates:
        fw_score = fuzz.token_sort_ratio(title, candidate)
        jaccard = jaccard_similarity(title, candidate)
        levenshtein = levenshtein_distance(title, candidate)

        if fw_score > best_fuzzy_score:
            best_fuzzy_score = fw_score
            best_match = candidate
            best_jaccard = jaccard
            best_levenshtein = levenshtein

            if best_fuzzy_score >= fuzzy_threshold or best_jaccard >= jaccard_threshold or best_levenshtein <= levenshtein_threshold or (best_fuzzy_score + best_jaccard * 100) >= 100:
                emb_title = encode_text([title])
                emb_candidate = encode_text([candidate])
                bert_score = util.pytorch_cos_sim(emb_title, emb_candidate).item()
                if bert_score > best_bert_score:
                    best_bert_score = bert_score

    return best_match, best_fuzzy_score, best_bert_score, best_jaccard, best_levenshtein

# מציאת כותרות תואמות בין 2017 ל-2018
matched_titles = {}
title_pairs = []
fuzzy_scores = []
bert_scores = []
jaccard_scores = []
levenshtein_scores = []

for title in titles_2017:
    best_match, fw_score, bert_score, jaccard, levenshtein = find_best_match(title, titles_2018)
    if best_match:
        matched_titles[title] = best_match
        title_pairs.append((title, best_match))
        fuzzy_scores.append(fw_score)
        bert_scores.append(bert_score)
        jaccard_scores.append(jaccard)
        levenshtein_scores.append(levenshtein)

# יצירת DataFrame עם ההתאמות
matched_df = pd.DataFrame(title_pairs, columns=["Title 2017", "Title 2018"])
matched_df["Fuzzy Score"] = fuzzy_scores
matched_df["BERT Similarity"] = bert_scores
matched_df["Jaccard Similarity"] = jaccard_scores
matched_df["Levenshtein Distance"] = levenshtein_scores

matched_df.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\LaBSE_Scores_Matched_Titles.xlsx", index=False)

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

df_unique_2017.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\LaBSE_Scores_Unique_Titles_2017.xlsx", index=False)
df_unique_2018.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\LaBSE_Scores_Unique_Titles_2018.xlsx", index=False)
df_common.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\LaBSE_Scores_Common_Titles.xlsx", index=False)

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

def compute_similarity(chunks_2017, chunks_2018, threshold=0.6):
    if not chunks_2017 or not chunks_2018:
        return 0, np.array([]), "", ""

    embeddings_2017 = encode_text(chunks_2017)
    embeddings_2018 = encode_text(chunks_2018)

    similarity_matrix = util.pytorch_cos_sim(embeddings_2017, embeddings_2018).cpu().numpy()

    best_matches = []
    worst_match_index = None
    worst_match_value = float("inf")

    for i in range(len(embeddings_2017)):
        best_match_index = np.argmax(similarity_matrix[i])  # מוצא את הפסקה עם ההתאמה הגבוהה ביותר
        best_similarity = similarity_matrix[i, best_match_index]
        best_matches.append(best_similarity)  # שומר את ההתאמות הטובות ביותר בלבד

        # מזהה את הפסקה שהותאמה הכי גרוע (כלומר עם הדמיון הנמוך ביותר מתוך ההתאמות שנבחרו)
        if best_similarity < worst_match_value:
            worst_match_value = best_similarity
            worst_match_index = (i, best_match_index)

    avg_similarity = np.mean(best_matches) if best_matches else 0  # חישוב ממוצע רק מההתאמות שנבחרו

    # אם אין התאמות, נחזיר מחרוזות ריקות
    if worst_match_index is None:
        return avg_similarity, similarity_matrix, "", ""

    return (
        avg_similarity,  # ממוצע הדמיון רק על ההתאמות הטובות ביותר
        similarity_matrix,
        chunks_2017[worst_match_index[0]],  # הפסקה עם ההתאמה הכי חלשה מתוך ההתאמות שנבחרו
        chunks_2018[worst_match_index[1]]
    )

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
df_similarity.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\LaBSE_Scores_Similarity_Results.xlsx", index=False)

print("תוצאות השוואת הטקסטים נשמרו בקובץ Similarity_Results.xlsx")

# הצגת סעיפים שהשתנו באופן משמעותי
print("\nסעיפים עם שינוי משמעותי:")
print(df_low_similarity)