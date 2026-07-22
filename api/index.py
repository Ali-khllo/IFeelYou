import os
import time
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="IFeelYou Emotion AI")

# Your custom fine-tuned model
MODEL_ID = "Alikhllo/IFeelYou-model"
HF_MODEL_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

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
        <title>IFeelYou - Emotion AI</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-white min-h-screen flex items-center justify-center p-4">
        <div class="max-w-md w-full bg-gray-800 rounded-xl p-6 shadow-2xl border border-gray-700">
            <h1 class="text-2xl font-bold mb-2 text-center text-indigo-400">IFeelYou Custom AI</h1>
            <p class="text-gray-400 text-sm mb-6 text-center">Powered by Alikhllo/IFeelYou-model</p>
            
            <textarea id="inputText" rows="4" class="w-full p-3 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 mb-4" placeholder="Type something here..."></textarea>
            
            <button id="btn" onclick="analyzeSentiment()" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2.5 rounded-lg transition duration-200">
                Analyze Emotion
            </button>
            
            <div id="result" class="mt-6 hidden p-4 rounded-lg bg-gray-900 border border-gray-700">
                <p class="text-xs text-gray-400">Result:</p>
                <div id="label" class="text-xl font-bold mt-1"></div>
                <div id="confidence" class="text-sm text-gray-400 mt-1 break-all"></div>
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
                labelDiv.innerText = "Warming up model...";
                labelDiv.className = "text-lg font-semibold mt-1 text-yellow-400";
                confDiv.innerText = "";
                btn.disabled = true;

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text })
                    });
                    const data = await res.json();
                    
                    if (data.label) {
                        const labelText = data.label.toUpperCase();
                        labelDiv.innerText = "Emotion: " + labelText;
                        
                        if (labelText.includes("HAPPY") || labelText.includes("JOY")) {
                            labelDiv.className = "text-xl font-bold mt-1 text-green-400";
                        } else if (labelText.includes("ANGER") || labelText.includes("SAD")) {
                            labelDiv.className = "text-xl font-bold mt-1 text-red-400";
                        } else {
                            labelDiv.className = "text-xl font-bold mt-1 text-indigo-400";
                        }
                        
                        const confPercent = data.confidence ? (data.confidence * 100).toFixed(2) + "%" : "N/A";
                        confDiv.innerText = "Confidence: " + confPercent;
                    } else if (data.error) {
                        labelDiv.innerText = data.error;
                        labelDiv.className = "text-base font-medium mt-1 text-red-400";
                        confDiv.innerText = data.details || "";
                    } else {
                        labelDiv.innerText = "Output:";
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
    if not HF_TOKEN:
        return {
            "error": "Configuration Error",
            "details": "HF_TOKEN environment variable is missing on Vercel."
        }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    # Retries for Cold Start on Hugging Face
    for attempt in range(3):
        try:
            response = requests.post(
                HF_MODEL_URL,
                headers=headers,
                json={"inputs": data.text},
                timeout=15
            )

            results = response.json()

            # If model is waking up, wait 3 seconds and retry automatically
            if isinstance(results, dict) and ("loading" in str(results) or "estimated_time" in str(results)):
                time.sleep(3)
                continue

            if response.status_code == 200:
                if isinstance(results, list) and len(results) > 0:
                    item = results[0]
                    if isinstance(item, list) and len(item) > 0:
                        top_result = item[0]
                        return {
                            "text": data.text,
                            "label": top_result.get("label"),
                            "confidence": top_result.get("score")
                        }
                    elif isinstance(item, dict) and "label" in item:
                        return {
                            "text": data.text,
                            "label": item.get("label"),
                            "confidence": item.get("score")
                        }
                return {"text": data.text, "raw_output": results}

            return {
                "error": f"Hugging Face Error ({response.status_code})",
                "details": str(results)
            }

        except Exception as e:
            if attempt == 2:
                return {"error": "Backend exception", "details": str(e)}
            time.sleep(2)