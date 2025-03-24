import os
import sqlite3
import pandas as pd
import requests
import nltk
import spacy
import cv2
import easyocr
import ffmpeg
import docx
from sentence_transformers import SentenceTransformer, util
from faster_whisper import WhisperModel

import nltk
import os
import zipfile
import shutil

def safe_nltk_download(resource):
    try:
        nltk.data.find(resource)
    except (LookupError, zipfile.BadZipFile):
        # Clean corrupted files if any
        resource_dir = os.path.join(os.path.expanduser("~"), "nltk_data", *resource.split("/"))
        if os.path.exists(resource_dir):
            shutil.rmtree(resource_dir, ignore_errors=True)
        nltk.download(resource.split("/")[-1])

# 🧠 Ensure these are downloaded
safe_nltk_download("tokenizers/punkt")
safe_nltk_download("corpora/stopwords")


spacy_model = spacy.load("en_core_web_sm")
spacy_model.max_length = 2000000

embedder = SentenceTransformer('all-MiniLM-L6-v2')
reader = easyocr.Reader(['en'])
whisper_model = WhisperModel("base", compute_type="float32")

TOGETHER_API_KEY = "a501f692103e0a408b02f73260c51d062359459f268acf87d36697c4ff82b0d6"

# === Database Setup ===
DB_FILE = "csv_data.db"

def init_db():
    if not os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_csv("Students_Grading_Dataset.csv")
        df.to_sql("students", conn, index=False, if_exists="replace")
        conn.close()

def get_csv_content_from_db():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM students", conn)
    conn.close()
    rows = []
    for _, row in df.iterrows():
        sentence = ", ".join([f"{col.strip()}: {str(row[col]).strip()}" for col in df.columns])
        rows.append(sentence)
    return rows

# === Loaders ===
def load_pdf(path):
    import pdfplumber
    if not os.path.exists(path):
        return "❌ PDF file not found."
    with pdfplumber.open(path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def load_docx(path):
    if not os.path.exists(path):
        return "❌ DOCX file not found."
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# === Preprocess ===
def preprocess(text):
    doc = spacy_model(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# === RAG ===
class SimpleRAG:
    def __init__(self):
        self.chunks = []
        self.embeddings = []

    def add_document(self, text, source):
        for sent in nltk.sent_tokenize(text):
            if sent.strip():
                cleaned = preprocess(sent)
                self.chunks.append((sent, source))
                self.embeddings.append(embedder.encode(cleaned))

    def retrieve(self, query, k=50):
        q_vec = embedder.encode(preprocess(query))
        scores = util.cos_sim(q_vec, self.embeddings)[0]
        top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [self.chunks[i][0] for i, _ in top_k]

# === Together API ===
def generate_answer_with_together(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 300
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No response from API.")

# === OCR & Audio ===
def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        return "❌ Image not found."
    result = reader.readtext(image_path)
    return " ".join([res[1] for res in result])

def extract_text_from_video(video_path, frame_interval=60):
    if not os.path.exists(video_path):
        return "❌ Video not found."
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "⚠️ Could not open video file."
    total_text = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            result = reader.readtext(frame)
            text = " ".join([res[1] for res in result])
            total_text.append(text)
        frame_count += 1
    cap.release()
    return " ".join(total_text)

def transcribe_audio(audio_or_video_path, is_video=True):
    audio_path = "temp_audio.wav" if is_video else audio_or_video_path
    try:
        if is_video:
            ffmpeg.input(audio_or_video_path).output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000').overwrite_output().run(quiet=True)
        segments, _ = whisper_model.transcribe(audio_path)
        transcript = " ".join([seg.text for seg in segments])
        if is_video:
            os.remove(audio_path)
        return transcript
    except Exception as e:
        return f"⚠️ Transcription failed: {e}"

# === Q&A ===
def ask_bot(query, kb, context):
    context_text = "\n".join(kb.retrieve(query)) if context is None else context
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context_text}\n\nQuestion: {query}"
    return generate_answer_with_together(prompt)

# === Main ===
if __name__ == "__main__":
    init_db()
    kb = SimpleRAG()

    pdf_path = "Data Science Interview.pdf"
    kb.add_document(load_pdf(pdf_path), "PDF")

    # Load CSV content from DB and embed
    for row in get_csv_content_from_db():
        kb.add_document(row, "CSV_DB")

    conversation_history = []

    print("\n✅ Chatbot ready! Ask questions or type 'exit' to quit.\n")

    while True:
        print("\n===== 💬 Past Conversation =====")
        if not conversation_history:
            print("No conversation history yet.")
        else:
            for i, (q, a) in enumerate(conversation_history, 1):
                print(f"{i}. 🧠 You: {q}\n   🤖 Bot: {a}")

        print("\n📁 Choose data source:")
        print("1. Text Query")
        print("2. Image")
        print("3. Video")
        print("4. Audio")
        print("5. DOCX")
        print("6. Exit")
        choice = input("Enter Choice: ").strip()
        full_context = None

        if choice == "1":
            user_q = input("🧠 You: ")
            if user_q.lower() in ("exit", "quit"):
                break
            response = ask_bot(user_q, kb, None)
        elif choice == "2":
            image_path = input("📷 Image path: ")
            full_context = extract_text_from_image(image_path)
            user_q = input("❓ Ask about image: ")
            response = ask_bot(user_q, kb, full_context)
        elif choice == "3":
            video_path = input("🎬 Video path: ")
            full_context = extract_text_from_video(video_path) + " " + transcribe_audio(video_path, is_video=True)
            user_q = input("❓ Ask about video: ")
            response = ask_bot(user_q, kb, full_context)
        elif choice == "4":
            audio_path = input("🎤 Audio path: ")
            full_context = transcribe_audio(audio_path, is_video=False)
            user_q = input("❓ Ask about audio: ")
            response = ask_bot(user_q, kb, full_context)
        elif choice == "5":
            docx_path = input("📘 DOCX file path: ")
            full_context = load_docx(docx_path)
            user_q = input("❓ Ask about DOCX: ")
            response = ask_bot(user_q, kb, full_context)
        elif choice == "6":
            break
        else:
            print("❌ Invalid choice.")
            continue

        conversation_history.append((user_q, response))
        print("🤖 Bot:", response)

