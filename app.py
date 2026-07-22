import os
import streamlit as st
from huggingface_hub import snapshot_download

MODEL_DIR = "model"

@st.cache_resource
def load_model():
    if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
        snapshot_download(
            repo_id="Alikhllo/IFeelYou-model",
            local_dir=MODEL_DIR,
        )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

model, tokenizer = load_model()