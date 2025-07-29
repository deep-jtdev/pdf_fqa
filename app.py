import gradio as gr
import uuid
import streamlit as st
import re
import PyPDF2
import numpy as np
from transformers import  AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import os
from database1 import create_db
from first1 import pdf_query

from ans_generator1 import AnswerGenerator

import  json
from q_generator1 import QGenerator
from transformers import pipeline
# Initialize models
qgen = QGenerator()
ansgen = AnswerGenerator()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
qa_model = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# ✅ Upload and process PDF
def upload_pdf(files):
    try:
        messages = []

        for fil in files:
            filename = os.path.basename(fil.name)  # <-- FIXED HERE
            token = str(uuid.uuid4())

            pdf_reader = PyPDF2.PdfReader(fil)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
            chunks = [text[i:i + 500] for i in range(0, len(text), 500)]

            create_db(token, chunks, filename)
            messages.append(f"✅ File Name: {filename}\n🔑 Token: {token}")

        return "\n\n".join(messages)

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ✅ Generate Questions & Answers
def generate_qa(token):
    try:
        if not token:
            return "⚠️ Please provide a token."

        print("📥 Received Token:", token)

        row=create_db.fetch_by_token_or_filename(token)
        print (row)
        if not row:
            return "❌ No data found for this token."

        chunks = json.loads(row['chunk_data'])
        qa_pairs = []

        for i, chunk in enumerate(chunks):
            print(f"\n🔹 Processing chunk {i+1}/{len(chunks)}")
            questions = qgen.generate(chunk)
            print(f"🧠 Questions generated: {questions}")

            if not questions:
                continue

            for question in questions[:1]:
                prompt = f"Context: {chunk}\n\nQuestion: {question}\n\nAnswer:"
                try:
                    result = qa_model(prompt, max_new_tokens=256, do_sample=False)
                    if isinstance(result, list) and "generated_text" in result[0]:
                        answer = result[0]["generated_text"].strip()
                    else:
                        answer = "N/A"

                    qa_pairs.append(f"Q: {question}\nA: {answer}")

                except Exception:
                    continue

        return "\n\n".join(qa_pairs) if qa_pairs else "⚠️ No Q&A pairs generated."

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ✅ Ask a question using token
def ask_question(token, question):
    try:
        row=create_db.fetch_by_token_or_filename(token)

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

# ✅ Gradio UI
with gr.Blocks(theme="default") as demo:
    gr.Markdown(
        """
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: #3b82f6;'>📄 AI-Powered PDF Q&A System</h1>
            <p style='font-size: 1.1rem;'>Upload your PDFs, generate smart questions, and get intelligent answers.</p>
        </div>
        """
    )

    with gr.Tab("📤 1. Upload PDF"):
        gr.Markdown("### 🗂 Upload a PDF File")
        file = gr.File(label="Upload one or more PDFs", file_types=[".pdf"], file_count="multiple")
        upload_out = gr.Textbox(label="Upload Result", interactive=False)
        file.change(fn=upload_pdf, inputs=file, outputs=upload_out)

    with gr.Tab("🧠 2. Generate Questions & Answers"):
        gr.Markdown("### 🤖 Generate Questions and Answers from Uploaded PDF")
        token_input = gr.Textbox(label="🔑 Enter Received Token", placeholder="e.g., 123e4567-e89b-12d3-a456...")
        output_box = gr.Textbox(label="📝 Generated Q&A", lines=15, interactive=False)
        gr.Button("🚀 Generate Q&A").click(fn=generate_qa, inputs=token_input, outputs=output_box)

    with gr.Tab("❓ 3. Ask a Question"):
        gr.Markdown("### 💬 Ask a question based on uploaded PDF")
        token_box = gr.Textbox(label="Token ID", placeholder="e.g., 123e4567-e89b-12d3-a456...")
        question_box = gr.Textbox(label="Type your question", placeholder="What is the main topic discussed?")
        answer_result = gr.Textbox(label="Answer Output", lines=6, interactive=False)
        gr.Button("🎯 Get Answer").click(fn=ask_question, inputs=[token_box, question_box], outputs=answer_result)

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860)

st.title("Hello Streamlit!")
st.write("This app is deployed from GitHub 🚀")
