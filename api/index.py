import os
import time
import requests
from fastapi import FastAPI
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
            
            <textarea id="inputText" rows="4" class="w-full p-3 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 mb-4" placeholder="Type something here..."></textarea>
            
            <button id="btn" onclick="analyzeSentiment()" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2.5 rounded-lg transition duration-200">
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
                const btn = document.getElementById('btn');

                resultDiv.classList.remove('hidden');
                labelDiv.innerText = "Analyzing... (Warming up AI model)";
                labelDiv.className = "text-lg font-semibold mt-1 text-yellow-400";
                confDiv.innerText = "This might take a few seconds on first request...";
                btn.disabled = true;

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text })
                    });
                    const data = await res.json();
                    
                    if (data.label) {
                        labelDiv.innerText = "Label: " + data.label;
                        labelDiv.className = data.label.toLowerCase().includes("pos") || data.label === "LABEL_1" 
                            ? "text-xl font-bold mt-1 text-green-400" 
                            : "text-xl font-bold mt-1 text-red-400";
                        
                        const confPercent = data.confidence ? (data.confidence * 100).toFixed(2) + "%" : "N/A";
                        confDiv.innerText = "Confidence: " + confPercent;
                    } else if (data.error) {
                        labelDiv.innerText = "Error: " + data.error;
                        labelDiv.className = "text-base font-medium mt-1 text-red-400";
                        confDiv.innerText = data.details || "";
                    } else {
                        labelDiv.innerText = "Unexpected output format";
                        confDiv.innerText = JSON.stringify(data);
                    }
                } catch (err) {
                    labelDiv.innerText = "Network Error";
                    confDiv.innerText = err.message;
                } finally {
                    btn.disabled = false;
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

    # Try calling HuggingFace up to 3 times to allow cold start
    for attempt in range(3):
        try:
            response = requests.post(
                HF_MODEL_URL,
                headers=headers,
                json={"inputs": data.text},
                timeout=15
            )
            
            if response.status_code == 200:
                results = response.json()
                
                # Format 1: [[{"label": "POSITIVE", "score": 0.98}, ...]]
                if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                    top_result = results[0][0]
                    return {
                        "text": data.text,
                        "label": top_result.get("label"),
                        "confidence": top_result.get("score")
                    }
                
                # Format 2: [{"label": "POSITIVE", "score": 0.98}]
                if isinstance(results, list) and len(results) > 0 and "label" in results[0]:
                    return {
                        "text": data.text,
                        "label": results[0].get("label"),
                        "confidence": results[0].get("score")
                    }
                
                return {"text": data.text, "raw_output": results}

            # If model is loading, wait and retry
            elif response.status_code in [503, 530]:
                time.sleep(4)
                continue
            else:
                return {
                    "error": f"Hugging Face API returned status {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            time.sleep(2)

    return {
        "error": "Model cold start timeout",
        "details": "The Hugging Face model took too long to wake up. Please try clicking Analyze again!"
    }