import streamlit as st
from transformers import pipeline
import PyPDF2

st.set_page_config(page_title="Smart Summary + Q&A", page_icon="🤖")

st.title("📘 AI Summarizer & Question Answering")

@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return summarizer, qa

summarizer, qa = load_models()

def summarize_large_text(text, chunk_size=1000):
    paragraphs = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        summary = summarizer(chunk, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
        paragraphs.append(summary)
    return " ".join(paragraphs)

uploaded_file = st.file_uploader("📤 Upload a PDF or text file", type=["pdf", "txt"])

text = ""
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        text = uploaded_file.read().decode("utf-8")

if text:
    st.subheader("🧾 Summarization")
    with st.spinner("Summarizing, please wait..."):
        summary = summarize_large_text(text)
    st.write(summary)
    st.download_button("⬇️ Download Summary", summary, "summary.txt")

    st.subheader("💬 Ask a Question")
    question = st.text_input("Type your question about the text:")
    if question:
        with st.spinner("Thinking..."):
            answer = qa(question=question, context=text)["answer"]
        st.success(f"**Answer:** {answer}")
