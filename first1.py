import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


class pdf_query:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.read = None

    def file(self, file):
        self.read = PyPDF2.PdfReader(file)

    def extract_text(self):
        text = ""
        for page in self.read.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text.strip()

    def split_into_chunks(self, text, chunk_size=300):
        # Split using punctuation for better sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def creat_model(self,chunks):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        chunk_embeddings = model.encode(chunks)
        return model,chunk_embeddings

    def answer_question(self,question, chunks, chunk_embeddings,model,threshold=0.6):
        q_embedding = model.encode([question])  # same model as above
        scores = cosine_similarity(q_embedding, chunk_embeddings)
        best_score = np.max(scores)
        best_chunk_index = np.argmax(scores)
        if best_score >= threshold:
            best_chunk = chunks[best_chunk_index]
            # Clean the answer
            cleaned_answer = re.sub(r'\s+', ' ', best_chunk.strip())
            return  cleaned_answer
        else:
            return {"answer": "Answer not found in PDF"}