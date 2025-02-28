import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# ✅ קבצי הקלט
file_2017_path = r"C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017_fixed.xlsx"
file_2018_path = r"C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018_fixed.xlsx"

# ✅ טעינת הנתונים
df_2017 = pd.read_excel(file_2017_path)
df_2018 = pd.read_excel(file_2018_path)

df_2017_clean = df_2017.dropna(subset=['hierarchy_level_name', 'content'])
df_2018_clean = df_2018.dropna(subset=['hierarchy_level_name', 'content'])

titles_2017 = list(df_2017_clean['hierarchy_level_name'])
titles_2018 = list(df_2018_clean['hierarchy_level_name'])

# ✅ רשימת המודלים לאימון
model_paths = {
    "DeBERTa": "microsoft/deberta-v3-large",
    #"DeBERTa": "microsoft/deberta-v3-large-finetuned-mnli",
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-large",
    "SimCSE": "princeton-nlp/sup-simcse-roberta-large"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")


# ✅ פונקציה ליצירת דאטה לאימון
class TaxDataset(Dataset):
    def __init__(self, titles_2017, titles_2018, tokenizer):
        self.pairs = []
        self.labels = []

        for title_2017 in titles_2017:
            best_match, fw_score, bert_score = self.find_best_match(title_2017, titles_2018)
            if best_match:
                self.pairs.append((title_2017, best_match))
                self.labels.append(bert_score)

        self.tokenizer = tokenizer

    def find_best_match(self, title, candidates):
        best_match, best_score, best_bert_score = None, 0, 0
        for candidate in candidates:
            fw_score = fuzz.token_sort_ratio(title, candidate)
            if fw_score > best_score:
                best_score, best_match = fw_score, candidate
                embeddings = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode(
                    [title, candidate], convert_to_tensor=True
                )
                bert_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                if bert_score > best_bert_score:
                    best_bert_score = bert_score
        return best_match, best_score, best_bert_score

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text1, text2 = self.pairs[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text1, text2, padding="max_length", truncation=True, max_length=256,
                                return_tensors="pt")
        return {key: val.squeeze(0) for key, val in inputs.items()}, torch.tensor(label, dtype=torch.float)


# ✅ Fine-Tuning לכל המודלים
for model_name, model_path in model_paths.items():
    print(f"\n🚀 Training {model_name}...")

    tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1).to(device)

    dataset = TaxDataset(titles_2017, titles_2018, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(dataloader) * 3
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    model.train()
    for epoch in range(3):
        for batch in dataloader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            loss = torch.nn.functional.mse_loss(outputs.logits.squeeze(), labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} completed for {model_name}.")

    # ✅ שמירת המודל המאומן
    fine_tuned_path = f"C:\\Users\\yifat\\PycharmProjects\\Final-Project\\Output_Files\\fine_tuned_{model_name}"
    model.save_pretrained(fine_tuned_path)
    tokenizer.save_pretrained(fine_tuned_path)
    print(f"✅ Fine-Tuned {model_name} saved to: {fine_tuned_path}")

    # ✅ השוואת כותרות עם המודל המאומן
    model.eval()
    results = []
    for title in titles_2017:
        best_match, fw_score, bert_score = dataset.find_best_match(title, titles_2018)
        results.append((title, best_match, fw_score, bert_score))

    df_results = pd.DataFrame(results, columns=["Title 2017", "Title 2018", "Fuzzy Score", "Fine-Tuned Similarity"])

    # ✅ שמירת תוצאות
    output_results_path = f"C:\\Users\\yifat\\PycharmProjects\\Final-Project\\Output_Files\\fine_tuned_results_{model_name}.xlsx"
    df_results.to_excel(output_results_path, index=False)
    print(f"📁 Fine-Tuned results for {model_name} saved to: {output_results_path}")

    # ✅ הצגת מדדים לאחר האימון
    mean_similarity = df_results["Fine-Tuned Similarity"].mean()
    accuracy = sum(df_results["Fine-Tuned Similarity"] > 0.8) / len(df_results)
    print(f"📊 {model_name} Fine-Tuned Metrics:\nMean Similarity: {mean_similarity:.4f}\nAccuracy: {accuracy:.4f}")
