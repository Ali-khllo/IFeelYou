import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="IFeelYou Sentiment API")

HF_REPO_ID = "Alikhllo/IFeelYou-model"

# Load tokenizer and PyTorch model directly from your Hugging Face repo
tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
model = AutoModelForSequenceClassification.from_pretrained(HF_REPO_ID)
model.eval()

class PredictRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "IFeelYou API running on Vercel"}

@app.post("/predict")
def predict(data: PredictRequest):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
    
    predicted_class = int(torch.argmax(probs, dim=-1)[0])
    confidence = float(probs[0][predicted_class])

    id2label = getattr(model.config, "id2label", {0: "NEGATIVE", 1: "POSITIVE"})
    label = id2label.get(predicted_class, f"LABEL_{predicted_class}")

    return {
        "text": data.text,
        "label": label,
        "confidence": confidence,
    }