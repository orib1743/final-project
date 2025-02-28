import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
import numpy as np
import os

# קבצי הקלט
file_2017_path = r"C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2017_fixed.xlsx"
file_2018_path = r"C:\Users\yifat\PycharmProjects\Final-Project\Data Collection & Preprocessing\Extracted_Hierarchy_Content_2018_fixed.xlsx"

# טעינת הנתונים
df_2017 = pd.read_excel(file_2017_path)
df_2018 = pd.read_excel(file_2018_path)

df_2017_clean = df_2017.dropna(subset=['hierarchy_level_name', 'content'])
df_2018_clean = df_2018.dropna(subset=['hierarchy_level_name', 'content'])

titles_2017 = list(df_2017_clean['hierarchy_level_name'])
titles_2018 = list(df_2018_clean['hierarchy_level_name'])

# שימוש במודל שעבר Fine-Tuning
model_paths = {
    "DeBERTa": "microsoft/deberta-v3-large-finetuned-mnli",  # מודל שעבר אימון קודם
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# פונקציה ליצירת דאטה לאימון
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


# Fine-Tuning עם מודל שעבר אימון קודם
for model_name, model_path in model_paths.items():
    print(f"\nTraining {model_name} with fine-tuned model...")

    tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1).to(device)

    dataset = TaxDataset(titles_2017, titles_2018, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # הקטנת batch size

    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(dataloader) * 2
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    model.train()
    step = 0
    for epoch in range(2):  # נוריד ל-2 epochs כדי לייעל זמן
        total_loss = 0
        num_batches = len(dataloader)

        for step, batch in enumerate(dataloader):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            loss = torch.nn.functional.mse_loss(outputs.logits.squeeze(), labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            # הדפסת ה-Loss כל 10 צעדים
            if step % 10 == 0:
                print(f"\nEpoch {epoch + 1}, Step {step}, Loss: {loss.item():.6f}")

            # שמירת המודל כל 50 צעדים
            if step % 50 == 0:
                save_path = f"C:\\Users\\yifat\\PycharmProjects\\Final-Project\\Output_Files\\checkpoint_{model_name}_epoch{epoch}_step{step}"
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"\nCheckpoint saved at step {step}")

        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch + 1} completed. Average Loss: {avg_loss:.6f}")

    # שמירת המודל המאומן
    fine_tuned_path = f"C:\\Users\\yifat\\PycharmProjects\\Final-Project\\Output_Files\\fine_tuned_{model_name}"
    model.save_pretrained(fine_tuned_path)
    tokenizer.save_pretrained(fine_tuned_path)
    print(f"\nFine-Tuned {model_name} saved to: {fine_tuned_path}")

    # שמירת תוצאות לאחר האימון
    output_results_path = f"C:\\Users\\yifat\\PycharmProjects\\Final-Project\\Output_Files\\fine_tuned_results_{model_name}.xlsx"
    print(f"\nFine-Tuned results for {model_name} saved to: {output_results_path}")
