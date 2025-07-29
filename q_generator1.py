from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# ✅ Use GPU if available, else CPU
device = 0 if torch.cuda.is_available() else -1

# ✅ Load model and tokenizer once (globally)
tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-hl", use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl")
qg_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

class QGenerator:
    def __init__(self):
        pass  # No need to reload model every time

    def split_sentences(self, text):
        # Very simple sentence splitting (you can use spaCy/NLTK for better results)
        return [s.strip() for s in text.split('.') if s.strip()]

    def chunk_text(self, text, chunk_size=512):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def generate(self, text, max_questions=1):
        questions = []
        sentences = self.split_sentences(text)

        if not sentences:
            return []

        # ✅ Only take required number of sentences
        batch_inputs = [f"generate question: {s} </s>" for s in sentences[:max_questions]]

        try:
            results = qg_pipeline(batch_inputs, max_new_tokens=128, num_return_sequences=1)
            for result in results:
                q = result.get("generated_text", "").strip()
                if q and q not in questions:
                    questions.append(q)
        except Exception as e:
            print("❌ Error generating questions:", e)

        return questions
