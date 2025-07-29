import os
import PyPDF2
from transformers import pipeline

UPLOAD_DIR = "uploaded_pdfs"

# ✅ Load the QA model ONCE globally (outside the class)
qa_pipeline = pipeline("question-answering", model="google/flan-t5-base")

#---------------------------------------------------------------
# updated the modal
# qa_pipeline = pipeline(
#     "question-answering",
#     model="tiiuae/falcon-7b-instruct",    # <-- Updated model here
#     tokenizer="tiiuae/falcon-7b-instruct"  # <-- Explicitly specifying tokenizer
# )
#---------------------------------------------------------------

class AnswerGenerator:
    def __init__(self):
        pass  # No need to load model here anymore

    def extract_pdf_text(self, token):
        pdf_path = os.path.join(UPLOAD_DIR, f"{token}.pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("PDF not found for given token")

        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return [page.extract_text() or "" for page in reader.pages]

    def generate_answers(self, token, questions):
        pages = self.extract_pdf_text(token)
        full_text = "\n".join(pages)
        results = []

        for question in questions:
            try:
                result = qa_pipeline(question=question, context=full_text)  # use global model
                results.append({"question": question, "answer": result["answer"]})
            except Exception as e:
                results.append({"question": question, "answer": "Error", "error": str(e)})

        return results