from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "./model"

app = FastAPI(title="IFeelYou")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Update this to match the label order your model was trained with
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


class TextInput(BaseModel):
    text: str


@app.get("/")
def root():
    return {"status": "ok", "message": "IFeelYou API is running"}


@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    top_idx = int(torch.argmax(probs))
    return {
        "text": input.text,
        "emotion": LABELS[top_idx] if top_idx < len(LABELS) else str(top_idx),
        "confidence": round(float(probs[top_idx]), 4),
        "all_scores": {
            LABELS[i] if i < len(LABELS) else str(i): round(float(p), 4)
            for i, p in enumerate(probs)
        },
    }