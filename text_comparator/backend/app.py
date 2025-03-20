from flask import Flask, request, jsonify
import os
import pandas as pd
import docx
import mammoth
import fitz  # PyMuPDF
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # עד 50MB

EXCEL_PATH = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\Output_Files\Similarity_Results_Fuzzy.xlsx'

def load_semantic_mapping(threshold=0.5):
    """ טוען את ההתאמות הסמנטיות מקובץ האקסל """
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")
        df = df.fillna("")  # ✅ המרת ערכים ריקים למחרוזת ריקה

        mappings = []

        for _, row in df.iterrows():
            similarity = row["Cosine Similarity"]
            text_2017 = str(row["Text 2017"]).strip()
            text_2018 = str(row["Text 2018"]).strip()

            if similarity >= threshold and text_2017 and text_2018:
                mappings.append((text_2017, text_2018))  # ✅ יצירת רשימה של התאמות

        print("📌 Mappings Loaded:", mappings)  # ✅ בדיקה שהנתונים נטענים
        return mappings
    except Exception as e:
        print(f"❌ שגיאה בטעינת האקסל: {e}")
        return []

def clean_text(text):
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])


from docx import Document


def read_docx(file_path):
    doc = Document(file_path)
    html_content = []

    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):  # ✅ זיהוי כותרות
            html_content.append(f"<h2>{para.text}</h2>")
        elif para.text.startswith("-") or para.text.startswith("•"):  # ✅ תבליטים
            html_content.append(f"<li>{para.text}</li>")
        else:
            html_content.append(f"<p>{para.text}</p>")  # ✅ שמירת פסקאות רגילות

    return "\n".join(html_content)


def read_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ הקובץ לא נמצא: {file_path}")
        return ""

    try:
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
    except Exception as e:
        print(f"❌ שגיאה בקריאת הקובץ {file_path}: {e}")
    return ""

def highlight_matching_paragraphs(text1, text2, mappings):
    """ מסמן את הפסקאות הדומות לפי המיפוי מהאקסל """
    highlighted_text1 = text1
    highlighted_text2 = text2

    for text_2017, text_2018 in mappings:
        if text_2017 in text1:
            print(f"✅ סימון פסקה במסמך 2017: {text_2017}")
            highlighted_text1 = highlighted_text1.replace(text_2017, f'<span class="semantic-match">{text_2017}</span>')

        if text_2018 in text2:
            print(f"✅ סימון פסקה במסמך 2018: {text_2018}")
            highlighted_text2 = highlighted_text2.replace(text_2018, f'<span class="semantic-match">{text_2018}</span>')

    return highlighted_text1, highlighted_text2

@app.route("/upload", methods=["POST"])
def upload_files():
    file1 = request.files["file1"]
    file2 = request.files["file2"]

    file1_path = os.path.join(app.config["UPLOAD_FOLDER"], file1.filename)
    file2_path = os.path.join(app.config["UPLOAD_FOLDER"], file2.filename)

    file1.save(file1_path)
    file2.save(file2_path)

    # קביעת הפונקציה לפי סוג הקובץ
    #if file1.filename.endswith(".txt"):
        #text1 = read_txt(file1_path)
    if file1.filename.endswith(".docx"):
        text1 = read_docx(file1_path)
    #elif file1.filename.endswith(".pdf"):
        #text1 = read_pdf(file1_path)
    else:
        text1 = "⚠️ Unsupported file type."

    #if file2.filename.endswith(".txt"):
        #text2 = read_txt(file2_path)
    if file2.filename.endswith(".docx"):
        text2 = read_docx(file2_path)
    #elif file2.filename.endswith(".pdf"):
        #text2 = read_pdf(file2_path)
    else:
        text2 = "⚠️ Unsupported file type."

    return jsonify({"text1": text1, "text2": text2})


if __name__ == "__main__":
    app.run(debug=True)
