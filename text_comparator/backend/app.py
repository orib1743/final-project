from flask import Flask, request, jsonify
import os
import pandas as pd
import docx
import mammoth
import fitz  # PyMuPDF
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # עד 50MB

EXCEL_PATH = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Output_Files\Similarity_Results_Fuzzy.xlsx'

def load_semantic_mapping(threshold=0.5):
    df = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")

    mappings = {}
    for _, row in df.iterrows():
        similarity = row["Cosine Similarity"]
        text_2017 = row["Text 2017"]
        text_2018 = row["Text 2018"]

        if similarity >= threshold:
            mappings[text_2017] = text_2018

    return mappings

def clean_text(text):
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def read_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ הקובץ לא נמצא: {file_path}")
        return ""

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            return clean_text(file.read())
    elif file_path.endswith(".docx"):
        with open(file_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            return clean_text(result.value)
    elif file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return clean_text(text)
    return ""

def highlight_matching_paragraphs(text1, text2, mappings):
    highlighted_text1 = []
    highlighted_text2 = []

    for para1 in text1.split("\n"):
        highlighted_text1.append(f'<span class="semantic-match">{para1}</span>' if para1 in mappings else para1)

    for para2 in text2.split("\n"):
        highlighted_text2.append(f'<span class="semantic-match">{para2}</span>' if para2 in mappings.values() else para2)

    return "<br>".join(highlighted_text1), "<br>".join(highlighted_text2)

@app.route("/upload", methods=["POST"])
def upload_files():
    file1 = request.files["file1"]
    file2 = request.files["file2"]

    text1 = read_file(file1)
    text2 = read_file(file2)
    mappings = load_semantic_mapping()

    highlighted_text1, highlighted_text2 = highlight_matching_paragraphs(text1, text2, mappings)
    return jsonify({"text1": highlighted_text1, "text2": highlighted_text2})

if __name__ == "__main__":
    app.run(debug=True)
