import streamlit as st
from streamlit_chat import message
import os
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# === Import backend functions ===
from main import (
    init_db, get_csv_content_from_db, load_pdf, load_docx,
    extract_text_from_image, extract_text_from_video, transcribe_audio,
    SimpleRAG, ask_bot
)

# === Streamlit Page Setup ===
st.set_page_config(page_title="Multi-Modal AI Chatbot", layout="wide")
st.title("🤖 Multi-Modal RAG Chatbot")

# === Initialize Knowledge Base (Once) ===
if "rag" not in st.session_state:
    with st.spinner("🔄 Initializing knowledge base... please wait."):
        init_db()
        kb = SimpleRAG()
        pdf_path = "Data Science Interview.pdf"
        with st.spinner("📘 Loading PDF..."):
            kb.add_document(load_pdf(pdf_path), "PDF")
        with st.spinner("🧠 Loading DB content..."):
            for row in get_csv_content_from_db():
                kb.add_document(row, "CSV_DB")
        st.session_state.rag = kb
        st.session_state.history = []

# === Sidebar: Mode Selection ===
with st.sidebar:
    st.header("📂 Select Data Source")
    data_mode = st.radio("Choose your input mode:", [
        "Text Query", "Image", "Video", "Audio", "DOCX"
    ])
    st.markdown("---")
    user_query = st.text_input("❓ Ask your question")
    uploaded_file = None

    if data_mode == "Image":
        uploaded_file = st.file_uploader("📷 Upload an image file", type=["png", "jpg", "jpeg"])
    elif data_mode == "Video":
        uploaded_file = st.file_uploader("🎬 Upload a video file", type=["mp4", "mov", "avi"])
    elif data_mode == "Audio":
        uploaded_file = st.file_uploader("🎤 Upload an audio file", type=["mp3", "wav", "m4a"])
    elif data_mode == "DOCX":
        uploaded_file = st.file_uploader("📘 Upload a DOCX file", type=["docx"])

# === Ask Button & Logic ===
if st.button("Ask") and user_query:
    context = None
    kb = st.session_state.rag

    if data_mode == "Text Query":
        context = None

    elif uploaded_file is not None:
        # Save file temporarily
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if data_mode == "Image":
            context = extract_text_from_image(temp_path)
        elif data_mode == "Video":
            context = extract_text_from_video(temp_path) + " " + transcribe_audio(temp_path, is_video=True)
        elif data_mode == "Audio":
            context = transcribe_audio(temp_path, is_video=False)
        elif data_mode == "DOCX":
            context = load_docx(temp_path)

    elif data_mode != "Text Query" and uploaded_file is None:
        st.warning("⚠️ Please upload a file to continue.")

    if data_mode == "Text Query" or context:
        answer = ask_bot(user_query, kb, context)
        st.session_state.history.append((user_query, answer))

# === Display Chat History ===
if st.session_state.history:
    st.subheader("💬 Chat History")
    for i, (q, a) in enumerate(st.session_state.history[::-1]):
        message(q, is_user=True, key=f"{i}_user")
        message(a, is_user=False, key=f"{i}_bot")
else:
    st.info("No chat history yet. Start by asking a question.")

# === Optional Debug Info ===
with st.expander("⚙️ Debug Info"):
    st.write("Knowledge Base Object:", st.session_state.rag)
    st.write("Chat History:", st.session_state.history)
