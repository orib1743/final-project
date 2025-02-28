from transformers import AutoModelForSequenceClassification, AutoTokenizer

save_directory = r"C:\Users\yifat\PycharmProjects\Models\deberta_finetuned"

# הורדת המודל החלופי
model_name = "cross-encoder/nli-deberta-v3-large"  # מודל שעבר Fine-Tuning למשימות NLI
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# שמירה מקומית
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"✅ Model saved to {save_directory}")
