from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "Ali-khllo/emotion-model"  # replace with your actual HF model repo name

app = FastAPI()

# Allow your Vercel frontend to call this API.
# Replace "*" with your actual Vercel domain once deployed, e.g. "https://your-app.vercel.app"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

id2label = model.config.id2label


class TextIn(BaseModel):
    text: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: TextIn):
    inputs = tokenizer(
        payload.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    scores = {id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}
    top_label = max(scores, key=scores.get)

    return {"label": top_label, "scores": scores}