import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="IFeelYou Sentiment API")

HF_MODEL_URL = "https://api-inference.huggingface.co/models/Alikhllo/IFeelYou-model"
HF_TOKEN = os.getenv("HF_TOKEN", "")

class PredictRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IFeelYou - Sentiment Analysis</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-white min-h-screen flex items-center justify-center p-4">
        <div class="max-w-md w-full bg-gray-800 rounded-xl p-6 shadow-2xl border border-gray-700">
            <h1 class="text-2xl font-bold mb-2 text-center text-indigo-400">IFeelYou Sentiment AI</h1>
            <p class="text-gray-400 text-sm mb-6 text-center">Analyze text emotion in real time</p>
            
            <textarea id="inputText" rows="4" class="w-full p-3 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 mb-4" placeholder="Type something here..."></type></textarea>
            
            <button onclick="analyzeSentiment()" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2.5 rounded-lg transition duration-200">
                Analyze Sentiment
            </button>
            
            <div id="result" class="mt-6 hidden p-4 rounded-lg bg-gray-900 border border-gray-700">
                <p class="text-xs text-gray-400">Result:</p>
                <div id="label" class="text-xl font-bold mt-1"></div>
                <div id="confidence" class="text-sm text-gray-400 mt-1"></div>
            </div>
        </div>

        <script>
            async function analyzeSentiment() {
                const text = document.getElementById('inputText').value;
                if (!text) return;

                const resultDiv = document.getElementById('result');
                const labelDiv = document.getElementById('label');
                const confDiv = document.getElementById('confidence');

                resultDiv.classList.remove('hidden');
                labelDiv.innerText = "Analyzing...";
                confDiv.innerText = "";

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text })
                    });
                    const data = await res.json();
                    
                    if (data.label) {
                        labelDiv.innerText = "Label: " + data.label;
                        labelDiv.className = data.label.includes("POS") ? "text-xl font-bold mt-1 text-green-400" : "text-xl font-bold mt-1 text-red-400";
                        confDiv.innerText = "Confidence: " + (data.confidence * 100).toFixed(2) + "%";
                    } else {
                        labelDiv.innerText = "Error processing request";
                    }
                } catch (err) {
                    labelDiv.innerText = "Failed to connect to API";
                }
            }
        </script>
    </body>
    </html>
    """

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
    
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
        top_result = results[0][0]
        return {
            "text": data.text,
            "label": top_result.get("label"),
            "confidence": top_result.get("score"),
            "raw_output": results[0]
        }

    return {"text": data.text, "result": results}