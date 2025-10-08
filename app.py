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

uploaded_file = st.file_uploader("📤 Upload a PDF or text file", type=["pdf", "txt"])

text = ""
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in reader.pages])
    else:
        text = uploaded_file.read().decode("utf-8")

if text:
    st.subheader("🧾 Summarization")
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    st.write(summary)

    st.subheader("💬 Ask a Question")
    question = st.text_input("Type your question about the text:")
    if question:
        answer = qa(question=question, context=text)["answer"]
        st.success(f"**Answer:** {answer}")
