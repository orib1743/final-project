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

# שמירת כותרות עם האינדקסים שלהן
df_2017_clean = df_2017_clean.reset_index(drop=True)
df_2018_clean = df_2018_clean.reset_index(drop=True)

df_2017_clean['index_2017'] = df_2017_clean.index + 1
df_2018_clean['index_2018'] = df_2018_clean.index + 1

titles_2017 = list(df_2017_clean[['index_2017', 'hierarchy_level_name']].itertuples(index=False, name=None))
titles_2018 = list(df_2018_clean[['index_2018', 'hierarchy_level_name']].itertuples(index=False, name=None))

# טען את מודל LaBSE
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE", do_lower_case=False)
model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def encode_text(texts):
    """ מקודד רשימה של טקסטים בעזרת LaBSE """
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings


def jaccard_similarity(str1, str2):
    """ מחשב דמיון Jaccard בין שתי מחרוזות לפי קבוצות מילים """
    set1, set2 = set(str1.split()), set(str2.split())
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

def find_best_match(title, candidate, fuzzy_threshold=80, jaccard_threshold=0.7, bert_weight=0.7, fuzzy_weight=0.2, jaccard_weight=0.1):
    """ מחפש את ההתאמה הטובה ביותר בעזרת BERT, Fuzzy ו-Jaccard עם עדיפות ל-BERT """
    fw_score = fuzz.token_sort_ratio(title, candidate)
    jaccard = jaccard_similarity(title, candidate)

    # חישוב ציון משולב ראשוני ללא BERT
    combined_score = fw_score * fuzzy_weight + jaccard * 100 * jaccard_weight  # Fuzzy ו-Jaccard עם משקל נמוך

    # חישוב ציון BERT
    emb_title = encode_text([title])
    emb_candidate = encode_text([candidate])
    bert_score = util.pytorch_cos_sim(emb_title, emb_candidate).item()

    # BERT מקבל את המשקל הגבוה ביותר
    combined_score += bert_score * 100 * bert_weight

    return candidate, fw_score, bert_score, jaccard  # אין צורך להחזיק 'best_match' כי יש רק התאמה אחת


# מציאת כותרות תואמות בין 2017 ל-2018
matched_titles = {}
title_pairs = []
fuzzy_scores = []
bert_scores = []
jaccard_scores = []
remaining_titles_2018 = titles_2018.copy()

# הדפסת 10 הכותרות הראשונות של 2018 לפני תחילת ההשוואות
print(f"🔹 Titles in 2018 before loop: {remaining_titles_2018[:10]}")

for index_2017, title_2017 in titles_2017:
    print(f"\n🔍 Processing {index_2017}/{len(titles_2017)}: {title_2017}")  # הצגת התקדמות

    if not remaining_titles_2018:
        break

    best_match = None
    best_fw_score = 0
    best_bert_score = 0
    best_jaccard = 0
    best_index_2018 = None

    for index_2018, candidate_2018 in remaining_titles_2018:

        # 🔍 בדיקה אם הכותרת "Subtitle A—Income Taxes" נבדקת מול מועמדים מ-2018
        if title_2017 == "Subtitle A—Income Taxes":
            print(f"⚡ Checking: {title_2017} (2017) vs. {candidate_2018} (2018)")

        match, fw_score, bert_score, jaccard = find_best_match(title_2017, candidate_2018)  # ✅ שליחת טקסט בלבד

        # 🔍 הדפסת תוצאות כל מדדי ההשוואה
        print(f"   🎯 Results -> Fuzzy: {fw_score}, Jaccard: {jaccard}, BERT: {bert_score}")

        # התאמה מתקבלת רק אם ה-BERT עובר את הסף וגם אחד הקריטריונים האחרים עובר סף
        if match and (bert_score >= 0.8 and (fw_score >= 80 or jaccard >= 0.7)):
            best_match = candidate_2018  # ✅ שמירת רק הכותרת
            best_fw_score = fw_score
            best_bert_score = bert_score
            best_jaccard = jaccard
            best_index_2018 = index_2018
            break  # אחרי שמצאנו התאמה טובה, נצא מהלולאה

    if best_match:
        matched_titles[index_2017] = best_index_2018
        print(f"✅ Matched: {title_2017} (2017) -> {best_match} (2018)")
        title_pairs.append((index_2017, title_2017, best_index_2018, best_match))
        fuzzy_scores.append(best_fw_score)
        bert_scores.append(best_bert_score)
        jaccard_scores.append(best_jaccard)

        # 🔍 בדיקה אם הכותרת מסולקת נכון מהרשימה
        print(f"❌ Removing {best_match} from remaining_titles_2018")
        remaining_titles_2018 = [(idx, t) for idx, t in remaining_titles_2018 if idx != best_index_2018]

    else:
        print(f"⚠️ Warning: No good match found for {title_2017} (2017)")


print("Matched Titles Dictionary:")
print(matched_titles)
print(f"Total matched titles: {len(matched_titles)} out of {len(titles_2017)}")

# יצירת DataFrame עם ההתאמות
matched_df = pd.DataFrame(title_pairs, columns=["Index 2017", "Title 2017", "Index 2018", "Title 2018"])
matched_df["Fuzzy Score"] = fuzzy_scores
matched_df["BERT Similarity"] = bert_scores
matched_df["Jaccard Similarity"] = jaccard_scores

matched_df.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\LaBSE_NoDup_Matched_Titles.xlsx",
                    index=False)

print("התאמות בין כותרות נשמרו בקובץ Matched_Titles.xlsx")

# עדכון סטים של כותרות משותפות וייחודיות עם מספור
df_matched_titles = matched_df[['Index 2017', 'Title 2017', 'Index 2018', 'Title 2018']]
common_titles = set(df_matched_titles["Title 2017"])  # כותרות משותפות שזוהו
unique_titles_2017 = df_2017_clean[~df_2017_clean['hierarchy_level_name'].isin(common_titles)]
unique_titles_2018 = df_2018_clean[~df_2018_clean['hierarchy_level_name'].isin(set(df_matched_titles["Title 2018"]))]

print(f"כותרות משותפות שנמצאו: {len(common_titles)}")
print(f"כותרות ייחודיות ל-2017: {len(unique_titles_2017)}")
print(f"כותרות ייחודיות ל-2018: {len(unique_titles_2018)}")

# יצירת DataFrame של כותרות ייחודיות ושמירתן לקובץ Excel
unique_titles_2017.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\LaBSE_NoDup_Unique_Titles_2017.xlsx", index=False)
unique_titles_2018.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\LaBSE_NoDup_Unique_Titles_2018.xlsx", index=False)
df_matched_titles.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\LaBSE_NoDup_Common_Titles.xlsx", index=False)

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
#content_2017 = {row['index_2017']: split_text(row['content']) for _, row in df_2017_clean.iterrows() if row['hierarchy_level_name'] in matched_df['Title 2017'].values}
#content_2018 = {row['index_2018']: split_text(row['content']) for _, row in df_2018_clean.iterrows() if row['hierarchy_level_name'] in matched_df['Title 2018'].values}
content_2017 = {row['index_2017']: split_text(row['content']) for _, row in df_2017_clean.iterrows() if row['index_2017'] in matched_df['Index 2017'].values}
content_2018 = {row['index_2018']: split_text(row['content']) for _, row in df_2018_clean.iterrows() if row['index_2018'] in matched_df['Index 2018'].values}

# פונקציה להשוואת דמיון תוכן בהתאמה לסדר הופעה
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
        best_matches.append(best_similarity)

        if best_similarity < worst_match_value:
            worst_match_value = best_similarity
            worst_match_index = (i, best_match_index)

    avg_similarity = np.mean(best_matches) if best_matches else 0

    if worst_match_index is None:
        return avg_similarity, similarity_matrix, "", ""

    return (
        avg_similarity,
        similarity_matrix,
        chunks_2017[worst_match_index[0]],
        chunks_2018[worst_match_index[1]]
    )

# חישוב הדמיון לכל סעיף בהתאמה סדרתית
similarities = []
titles_common = []
detail_changes = []
changed_text_2017 = []
changed_text_2018 = []

for _, row in matched_df.iterrows():
    #index_2017, title_2017, index_2018, title_2018 = row
    index_2017 = row["Index 2017"]
    title_2017 = row["Title 2017"]
    index_2018 = row["Index 2018"]
    title_2018 = row["Title 2018"]

    sim, sim_matrix, text_2017, text_2018 = compute_similarity(content_2017.get(index_2017, []), content_2018.get(index_2018, []))
    similarities.append(sim)
    titles_common.append(title_2017)
    min_sim = np.min(sim_matrix) if sim_matrix.size > 0 else 0
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
df_similarity.to_excel(r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files\LaBSE_NoDup_Similarity_Results.xlsx", index=False)

print("תוצאות השוואת הטקסטים נשמרו בקובץ Similarity_Results.xlsx")

# הצגת סעיפים שהשתנו באופן משמעותי
print("\nסעיפים עם שינוי משמעותי:")
print(df_low_similarity)
