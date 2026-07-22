import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="IFeelYou Sentiment API")

HF_MODEL_URL = "https://api-inference.huggingface.co/models/Alikhllo/IFeelYou-model"
# Optional: Set HF_TOKEN in Vercel Environment Variables if repo is private
HF_TOKEN = os.getenv("HF_TOKEN", "")

class PredictRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "IFeelYou API running on Vercel"}

@app.post("/predict")
def predict(data: PredictRequest):
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    response = requests.post(
        HF_MODEL_URL,
        headers=headers,
        json={"inputs": data.text},
        timeout=10
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Hugging Face API error: {response.text}"
        )

    results = response.json()
    
    # HF Inference API returns list of candidate labels with scores
    # e.g., [[{"label": "POSITIVE", "score": 0.98}, {"label": "NEGATIVE", "score": 0.02}]]
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
        top_result = results[0][0]
        return {
            "text": data.text,
            "label": top_result.get("label"),
            "confidence": top_result.get("score"),
            "raw_output": results[0]
        }

    return {"text": data.text, "result": results}