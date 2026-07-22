import os
import requests
import streamlit as st

MODEL_DIR = "model"
REPO_ID = "Alikhllo/IFeelYou-model"
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]

@st.cache_resource
def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

    for filename in FILES:
        local_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(local_path):
            url = f"https://huggingface.co/{REPO_ID}/resolve/main/{filename}"
            r = requests.get(url, headers=headers, stream=True)
            if r.status_code != 200:
                st.error(f"Failed on {filename}: status {r.status_code} — {r.text[:300]}")
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

model, tokenizer = load_model()