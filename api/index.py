import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

app = FastAPI(title="IFeelYou Sentiment API")

HF_REPO_ID = "Alikhllo/IFeelYou-model"

# Download ONNX model files from HF Hub into temp storage on function start
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model.onnx")
try:
    hf_hub_download(repo_id=HF_REPO_ID, filename="model.onnx.data")
except Exception:
    pass

tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
session = ort.InferenceSession(model_path)

class PredictRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "IFeelYou API running on Vercel"}

@app.post("/predict")
def predict(data: PredictRequest):
    inputs = tokenizer(data.text, return_tensors="np", truncation=True, padding=True)
    
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    logits = session.run(None, ort_inputs)[0]
    
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    predicted_class = int(np.argmax(probs, axis=-1)[0])
    confidence = float(probs[0][predicted_class])

    id2label = getattr(tokenizer, "id2label", {0: "NEGATIVE", 1: "POSITIVE"})
    label = id2label.get(predicted_class, f"LABEL_{predicted_class}")
    
    return {
        "text": data.text,
        "label": label,
        "confidence": confidence,
    }