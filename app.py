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


st.title("IFeelYou — Emotion Detector")
st.write("Enter some text and I'll tell you what emotion it expresses.")

with st.spinner("Loading model..."):
    model, tokenizer = load_model()

text = st.text_area("Your text:")

if st.button("Analyze"):
    if text.strip():
        import torch
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        labels = model.config.id2label
        top_idx = probs.argmax().item()
        st.success(f"Predicted emotion: **{labels[top_idx]}** ({probs[top_idx]*100:.1f}% confidence)")

        st.write("All scores:")
        for i, p in enumerate(probs):
            st.write(f"{labels[i]}: {p*100:.1f}%")
    else:
        st.warning("Please enter some text first.")