import streamlit as st
import uuid
import re
import PyPDF2
import numpy as np
import json
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity

from database1 import create_db
from first1 import pdf_query
from ans_generator1 import AnswerGenerator
from q_generator1 import QGenerator

# Load models
qgen = QGenerator()
ansgen = AnswerGenerator()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
qa_model = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Upload PDF
def upload_pdf(files):
    try:
        messages = []
        for fil in files:
            filename = os.path.basename(fil.name)
            token = str(uuid.uuid4())
            pdf_reader = PyPDF2.PdfReader(fil)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
            chunks = [text[i:i + 500] for i in range(0, len(text), 500)]

            create_db(token, chunks, filename)
            messages.append(f"✅ File Name: {filename}\n🔑 Token: {token}")
        return "\n\n".join(messages)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Generate Q&A
def generate_qa(token):
    try:
        if not token:
            return "⚠️ Please provide a token."

        row = create_db.fetch_by_token_or_filename(token)
        if not row:
            return "❌ No data found for this token."

        chunks = json.loads(row['chunk_data'])
        qa_pairs = []

        for chunk in chunks:
            questions = qgen.generate(chunk)
            if not questions:
                continue
            for question in questions[:1]:  # only 1 question per chunk
                prompt = f"Context: {chunk}\n\nQuestion: {question}\n\nAnswer:"
                result = qa_model(prompt, max_new_tokens=256, do_sample=False)
                if isinstance(result, list) and "generated_text" in result[0]:
                    answer = result[0]["generated_text"].strip()
                else:
                    answer = "N/A"
                qa_pairs.append(f"Q: {question}\nA: {answer}")
        return "\n\n".join(qa_pairs) if qa_pairs else "⚠️ No Q&A pairs generated."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Ask question
def ask_question(token, question):
    try:
        row = create_db.fetch_by_token_or_filename(token)
        if not row:
            return "❌ Token not found."

        chunks = json.loads(row['chunk_data'])
        processor = pdf_query()
        model = processor.model

        clean_chunks = [re.sub(r'\s+', ' ', c.strip()) for c in chunks if c.strip()]
        if not clean_chunks:
            return "⚠️ No valid content found in PDF."

        chunk_embeddings = model.encode(clean_chunks)
        q_embedding = model.encode([question])
        scores = cosine_similarity(q_embedding, chunk_embeddings)[0]

        top_index = int(np.argmax(scores))
        top_score = float(scores[top_index])
        best_text = clean_chunks[top_index]

        return f"Q: {question}\nA: {best_text}\nScore: {round(top_score, 3)}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="PDF Q&A AI System", layout="wide")
st.title("📄 AI-Powered PDF Q&A System")

tab1, tab2, tab3 = st.tabs(["📤 Upload PDF", "🧠 Generate Q&A", "❓ Ask a Question"])

with tab1:
    st.header("Upload one or more PDF files")
    uploaded_files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Upload PDFs"):
        if uploaded_files:
            upload_result = upload_pdf(uploaded_files)
            st.success(upload_result)
        else:
            st.warning("Please upload at least one PDF.")

with tab2:
    st.header("Generate Questions and Answers from PDF")
    token_input = st.text_input("Enter token from uploaded PDF")
    if st.button("Generate Q&A"):
        result = generate_qa(token_input.strip())
        st.text_area("Generated Questions & Answers", result, height=300)

with tab3:
    st.header("Ask a Question Based on PDF")
    token = st.text_input("Token for lookup")
    question = st.text_input("Enter your question")
    if st.button("Get Answer"):
        output = ask_question(token.strip(), question.strip())
        st.text_area("Answer", output, height=200)
