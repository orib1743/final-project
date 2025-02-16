import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import os

import openai
print("OpenAI module is installed correctly!")

# טען את הנתונים
file_2017_path = r"C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017_fixed.xlsx"
file_2018_path = r"C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018_fixed.xlsx"

df_2017 = pd.read_excel(file_2017_path)
df_2018 = pd.read_excel(file_2018_path)

# נקה נתונים ריקים
df_2017_clean = df_2017.dropna(subset=['hierarchy_level_name', 'content'])
df_2018_clean = df_2018.dropna(subset=['hierarchy_level_name', 'content'])

titles_2017 = list(df_2017_clean['hierarchy_level_name'])
titles_2018 = list(df_2018_clean['hierarchy_level_name'])

# בחר מודלים להשוואה
models = {
    "Agnostic BERT": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
    "RoBERTa": SentenceTransformer("sentence-transformers/all-roberta-large-v1"),
    #"DeBERTa": SentenceTransformer("sentence-transformers/deberta-v3-base"),
    #"DeBERTa": SentenceTransformer("microsoft/deberta-v3-small"),
    "DeBERTa": SentenceTransformer("microsoft/deberta-v3-large"),
    "SimCSE": SentenceTransformer("princeton-nlp/sup-simcse-roberta-large")
}


def find_best_match(title, candidates, model, threshold=80):
    """ מחפש את ההתאמה הטובה ביותר לפי fuzzywuzzy ואז בודק התאמה באמצעות המודל שנבחר """
    best_match, best_score, best_bert_score = None, 0, 0

    for candidate in candidates:
        fw_score = fuzz.token_sort_ratio(title, candidate)
        if fw_score > best_score:
            best_score, best_match = fw_score, candidate

            if best_score >= threshold:
                embeddings = model.encode([title, candidate], convert_to_tensor=True)
                bert_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                if bert_score > best_bert_score:
                    best_bert_score = bert_score

    return best_match, best_score, best_bert_score


def evaluate_model(model_name, model):
    matched_titles = {}
    title_pairs, fuzzy_scores, bert_scores, gpt_differences = [], [], [], []

    for title in titles_2017:
        best_match, fw_score, bert_score = find_best_match(title, titles_2018, model)
        if best_match:
            matched_titles[title] = best_match
            title_pairs.append((title, best_match))
            fuzzy_scores.append(fw_score)
            bert_scores.append(bert_score)

            # השוואת טקסטים באמצעות GPT-4
            text_2017 = df_2017_clean[df_2017_clean['hierarchy_level_name'] == title]['content'].values[0]
            text_2018 = df_2018_clean[df_2018_clean['hierarchy_level_name'] == best_match]['content'].values[0]
            gpt_difference = compare_texts_with_gpt(text_2017, text_2018)
            gpt_differences.append(gpt_difference)

    df_results = pd.DataFrame(title_pairs, columns=["Title 2017", "Title 2018"])
    df_results["Fuzzy Score"] = fuzzy_scores
    df_results["BERT Similarity"] = bert_scores
    df_results["GPT Differences"] = gpt_differences

    return df_results


# הוספת השוואה באמצעות GPT-4
openai.api_key = os.getenv("OPENAI_API_KEY")  # יש להגדיר את ה-API key בסביבת המשתמש


def compare_texts_with_gpt(text1, text2):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in legal text comparison."},
            {"role": "user",
             "content": f"Compare the following legal texts:\n\nText 1: {text1}\n\nText 2: {text2}\n\nSummarize the main differences."}
        ]
    )
    return response["choices"][0]["message"]["content"]


# הרצת כל המודלים והשוואת התוצאות
results = {}
for model_name, model in models.items():
    print(f"Running model: {model_name}")
    results[model_name] = evaluate_model(model_name, model)

# ניתוח תוצאות
metrics = {}
for model_name, df in results.items():
    mean_similarity = df["BERT Similarity"].mean()
    std_dev = df["BERT Similarity"].std()
    accuracy = sum(df["BERT Similarity"] > 0.8) / len(df)  # נחשב דיוק לפי סף של 0.8
    precision = precision_score(df["BERT Similarity"] > 0.8, [True] * len(df), zero_division=0)
    recall = recall_score(df["BERT Similarity"] > 0.8, [True] * len(df), zero_division=0)
    f1 = f1_score(df["BERT Similarity"] > 0.8, [True] * len(df), zero_division=0)

    metrics[model_name] = {
        "Mean Similarity": mean_similarity,
        "Std Dev": std_dev,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# הצגת תוצאות בטבלה
metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
print(metrics_df)

# הצגת גרף השוואתי
metrics_df.plot(kind='bar', figsize=(12, 6), title='Model Comparison')
plt.xticks(rotation=45)
plt.ylabel('Score')
plt.show()
