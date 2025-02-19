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


import openai
print("OpenAI module is installed correctly!")
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key found: {api_key[:5]}...")  # הדפסת חלק מהמפתח לבדיקה

CACHE_FILE = "gpt_cache.json"

def load_cache():
    """ טוען את קובץ ה-Cache אם הוא קיים """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """ שומר את ה-Cache לקובץ JSON """
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

# טען את ה-Cache בתחילת הריצה
gpt_cache = load_cache()


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

#model_name_mistral = "mistralai/Mistral-7B-Instruct-v0.2"
model_name_mistral = "open-mistral/mistral-7b"
mistral_tokenizer = AutoTokenizer.from_pretrained(model_name_mistral)
mistral_model = AutoModelForCausalLM.from_pretrained(model_name_mistral, torch_dtype="auto", device_map="auto")

# הגדרת המודלים להשוואה
models = {
    "Agnostic BERT": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
    "RoBERTa": SentenceTransformer("sentence-transformers/all-roberta-large-v1"),
    "DeBERTa": (deberta_model, deberta_tokenizer),  # שמור את המודל וה-tokenizer כטאפלה
    "SimCSE": SentenceTransformer("princeton-nlp/sup-simcse-roberta-large")
}

def compare_texts_with_mistral(text1, text2):
    """
    השוואת טקסטים באמצעות Mistral 7B-Instruct
    """
    prompt = f"Compare the following legal texts:\n\nText 1: {text1}\n\nText 2: {text2}\n\nSummarize the main differences:"

    inputs = mistral_tokenizer(prompt, return_tensors="pt").to(
        "cuda" if torch.cuda.is_available() else "cpu")  # המרה ל-GPU אם אפשרי
    with torch.no_grad():
        #outputs = mistral_model.generate(**inputs, max_new_tokens=300, do_sample=True, top_k=50, temperature=0.7)
        outputs = mistral_model.generate(**inputs, max_new_tokens=50, do_sample=True, top_k=50, temperature=0.7)

    result = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


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
    title_pairs, fuzzy_scores, bert_scores, mistral_differences = [], [], [], []

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

            # השוואת טקסטים באמצעות Mistral
            text_2017 = df_2017_clean[df_2017_clean['hierarchy_level_name'] == title]['content'].values[0]
            text_2018 = df_2018_clean[df_2018_clean['hierarchy_level_name'] == best_match]['content'].values[0]
            mistral_difference = compare_texts_with_mistral(text_2017, text_2018)
            mistral_differences.append(mistral_difference)

            # השוואת טקסטים באמצעות GPT-4
            #text_2017 = df_2017_clean[df_2017_clean['hierarchy_level_name'] == title]['content'].values[0]
            #text_2018 = df_2018_clean[df_2018_clean['hierarchy_level_name'] == best_match]['content'].values[0]
            #gpt_difference = compare_texts_with_gpt(text_2017, text_2018)
            #gpt_differences.append(gpt_difference)

    df_results = pd.DataFrame(title_pairs, columns=["Title 2017", "Title 2018"])
    df_results["Fuzzy Score"] = fuzzy_scores
    df_results["BERT Similarity"] = bert_scores
    df_results["Mistral Differences"] = mistral_differences
    #df_results["GPT Differences"] = gpt_differences

    return df_results


# הוספת השוואה באמצעות GPT-4
def compare_texts_with_gpt(text1, text2):
    api_key = os.getenv("OPENAI_API_KEY")  # קבלת המפתח מהסביבה
    if not api_key:
        raise ValueError("❌ Missing OpenAI API key! Set it in the environment or pass it explicitly.")

    client = openai.OpenAI(api_key=api_key)

    # יצירת מפתח Hash כדי לזהות השוואות ייחודיות
    text_hash = hashlib.md5(f"{text1}||{text2}".encode()).hexdigest()

    # בדיקה אם התוצאה כבר קיימת ב-Cache
    if text_hash in gpt_cache:
        print(f"📌 Using cached result for texts.")
        return gpt_cache[text_hash]

    # אם התוצאה לא קיימת, שליחת בקשה ל-GPT
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # עדכון השם מ-gpt-4 ל-gpt-4o
            messages=[
                {"role": "system", "content": "You are an expert in legal text comparison."},
                {"role": "user",
                 "content": f"Compare the following legal texts:\n\nText 1: {text1}\n\nText 2: {text2}\n\nSummarize the main differences."}
            ]
        )
        result = response.choices[0].message.content

        # שמירת התוצאה ב-Cache
        gpt_cache[text_hash] = result
        save_cache(gpt_cache)

        return result

    except openai.RateLimitError as e:
        print(f"🚨 Rate Limit Error: {e}")
        return "Error: Rate Limit Exceeded"
    except openai.OpenAIError as e:
        print(f"❌ OpenAI API Error: {e}")
        return "Error: API Request Failed"


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
# עדכון ניתוח התוצאות כך ש-GPT יהיה כלול
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

    mistral_diff_length = df["Mistral Differences"].apply(lambda x: len(str(x)) if isinstance(x, str) else 0).mean()

    # נחשב את מידת ההבדל על בסיס אורך הטקסט שמוחזר מהשוואת GPT (רק אם יש GPT Differences)
    #if "GPT Differences" in df.columns:
    #    gpt_diff_length = df["GPT Differences"].apply(lambda x: len(str(x)) if isinstance(x, str) else 0).mean()
    #else:
    #    gpt_diff_length = None  # או להגדיר ערך ברירת מחדל כמו 0

    metrics[model_name] = {
        "Mean Similarity": mean_similarity,
        "Std Dev": std_dev,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Mistral Diff Length": mistral_diff_length
        #"GPT Diff Length": gpt_diff_length  # כמה הטקסטים שונים לפי GPT
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

    # הפרדה בין המדדים של הדיוק והמדדים של ההבדלים של GPT
    #gpt_metric_exists = "GPT Diff Length" in metrics_df.columns

    mistral_metric_exists = "Mistral Diff Length" in metrics_df.columns

    # הצגת מדדי דיוק ודמיון (ללא GPT Diff Length)
    #metrics_no_gpt = metrics_df.drop(columns=["GPT Diff Length"]) if gpt_metric_exists else metrics_df
    metrics_no_mistral = metrics_df.drop(columns=["Mistral Diff Length"]) if mistral_metric_exists else metrics_df
    #metrics_no_gpt.plot(kind='bar', figsize=(12, 6), title='Model Comparison')
    metrics_no_mistral.plot(kind='bar', figsize=(12, 6), title='Model Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Score')
    plt.show()

    # אם יש מדדי הבדל של GPT, נציג אותם בנפרד
    if mistral_metric_exists:  # gpt_metric_exists:
        #metrics_df["GPT Diff Length"].plot(kind='bar', figsize=(12, 6), color='red', title='GPT Difference Length')
        metrics_df["Mistral Diff Length"].plot(kind='bar', figsize=(12, 6), color='red', title='Mistral Diff Length')
        plt.xticks(rotation=45)
        plt.ylabel('Difference Length')
        plt.show()

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
