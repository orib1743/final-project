import pandas as pd
#from transformers import DebertaV2Tokenizer, AutoModel
from transformers import DebertaV2Tokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import json
import hashlib
import torch

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


# טען את המודל וה-tokenizer
deberta_tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
deberta_model = AutoModel.from_pretrained("microsoft/deberta-v3-large")

# ✅ בדיקה שהמודל אכן נטען ל-GPU
import torch
print(f"🔍 GPU available: {torch.cuda.is_available()}")
print(f"🔢 Number of GPUs: {torch.cuda.device_count()}")
print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
#print(f"🚀 Model is on: {mistral_model.device}")

# הגדרת המודלים להשוואה
models = {
    "Agnostic BERT": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
    "RoBERTa": SentenceTransformer("sentence-transformers/all-roberta-large-v1"),
    "DeBERTa": (deberta_model, deberta_tokenizer),  # שמור את המודל וה-tokenizer כטאפלה
    "SimCSE": SentenceTransformer("princeton-nlp/sup-simcse-roberta-large")
}


def find_best_match(title, candidates, model, tokenizer=None, threshold=80):
    """ מחפש את ההתאמה הטובה ביותר לפי fuzzywuzzy ואז בודק התאמה באמצעות המודל שנבחר """
    best_match, best_score, best_bert_score = None, 0, 0

    for candidate in candidates:
        fw_score = fuzz.token_sort_ratio(title, candidate)
        if fw_score > best_score:
            best_score, best_match = fw_score, candidate

            if best_score >= threshold:
                if tokenizer:  # אם זה DeBERTa
                    inputs = tokenizer([title, candidate], return_tensors="pt", padding=True, truncation=True)
                    embeddings = model(**inputs).last_hidden_state.mean(dim=1)
                else:
                    embeddings = model.encode([title, candidate], convert_to_tensor=True)

                bert_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                if bert_score > best_bert_score:
                    best_bert_score = bert_score

    return best_match, best_score, best_bert_score

def evaluate_model(model_name, model):
    matched_titles = {}
    title_pairs, fuzzy_scores, bert_scores = [], [], []

    # בדיקה אם זה מודל DeBERTa (כי הוא טוען בצורה שונה)
    if isinstance(model, tuple):
        model, tokenizer = model  # חלץ את המודל וה-tokenizer
    else:
        tokenizer = None

    for title in titles_2017:
        best_match, fw_score, bert_score = find_best_match(title, titles_2018, model, tokenizer)
        if best_match:
            matched_titles[title] = best_match
            title_pairs.append((title, best_match))
            fuzzy_scores.append(fw_score)
            bert_scores.append(bert_score)

    df_results = pd.DataFrame(title_pairs, columns=["Title 2017", "Title 2018"])
    df_results["Fuzzy Score"] = fuzzy_scores
    df_results["BERT Similarity"] = bert_scores

    return df_results

# הרצת כל המודלים והשוואת התוצאות
results = {}
for model_name, model in models.items():
    print(f"Running model: {model_name}")
    df_result = evaluate_model(model_name, model)

    # אם df_result ריק (למשל, DeBERTa נכשל), נדלג עליו
    if df_result.empty:
        print(f"⚠️ Warning: No results for {model_name}. Skipping.")
        continue

    results[model_name] = df_result

# ניתוח תוצאות
metrics = {}
for model_name, df in results.items():
    if "BERT Similarity" not in df.columns:
        print(f"⚠️ Warning: {model_name} does not have 'BERT Similarity', skipping metric calculations.")
        continue  # מדלגים על מודלים שלא מחזירים דמיון ישיר

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

# הצגת גרף
metrics_df.plot(kind='bar', figsize=(12, 6), title='Model Comparison')
plt.xticks(rotation=45)
plt.ylabel('Score')
plt.show()

# בדיקה אם יש נתונים להציג
if not metrics_df.empty:
    print(metrics_df)
else:
    print("⚠️ No valid metrics to display.")


def export_results(metrics_df, results, output_dir=r"C:\Users\yifat\PycharmProjects\Final-Project\Output_Files"):
    """
    שומר את תוצאות המדדים וההשוואה בין המודלים לקובץ Excel ומייצא את הגרף כתמונה
    :param metrics_df: DataFrame שמכיל את המדדים הסופיים לכל מודל
    :param results: מילון שמכיל את תוצאות ההתאמה לכל מודל
    :param output_dir: תיקיית היעד לשמירת הקבצים
    """

    # יצירת תיקייה אם היא לא קיימת
    os.makedirs(output_dir, exist_ok=True)

    # שמירת המדדים בטבלה
    metrics_file = os.path.join(output_dir, "model_metrics.xlsx")
    metrics_df.to_excel(metrics_file)
    print(f"📁 Metrics saved to: {metrics_file}")

    # שמירת התוצאות של כל מודל
    for model_name, df in results.items():
        results_file = os.path.join(output_dir, f"{model_name}_results.xlsx")
        df.to_excel(results_file, index=False)
        print(f"📁 {model_name} results saved to: {results_file}")

    # שמירת הגרף כתמונה
    graph_file = os.path.join(output_dir, "model_comparison.png")
    metrics_df.plot(kind='bar', figsize=(12, 6), title='Model Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Score')
    plt.savefig(graph_file)  # שמירה לקובץ תמונה
    print(f"📊 Graph saved to: {graph_file}")

    print("✅ Export completed successfully!")


# קריאה לפונקציה לאחר הצגת התוצאות
if not metrics_df.empty:
    export_results(metrics_df, results)
