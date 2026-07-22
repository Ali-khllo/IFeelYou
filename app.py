import streamlit as st
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
import random
import json
import time
import os

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="EmotionSense",
    page_icon="🌈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "model.onnx"
TOKENIZER_DIR = "tokenizer_files"
MAX_LEN = 128

# ----------------------------
# Styling
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 20%, #1a1c2e 0%, #0e0f1a 55%, #08090f 100%);
}

#MainMenu, footer, header {visibility: hidden;}

.hero {
    text-align: center;
    padding: 1.2rem 0 0.4rem 0;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #f472b6, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero p {
    color: #9ca3af;
    font-size: 1.02rem;
    margin-top: 0;
}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.6rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    color: #f3f4f6 !important;
    font-size: 1.02rem !important;
    padding: 14px !important;
}
.stTextArea textarea:focus {
    border: 1px solid #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
}

.stButton>button {
    width: 100%;
    border-radius: 14px;
    border: none;
    padding: 0.7rem 1rem;
    font-weight: 700;
    font-size: 1.02rem;
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
    color: white;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(139, 92, 246, 0.35);
}

.result-emotion {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.4rem;
}
.result-emotion .emoji {
    font-size: 2.6rem;
}
.result-emotion .label {
    font-size: 1.7rem;
    font-weight: 800;
    text-transform: capitalize;
}
.result-conf {
    color: #9ca3af;
    font-size: 0.92rem;
    margin-bottom: 1rem;
}

.bar-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.45rem;
}
.bar-label {
    width: 90px;
    font-size: 0.85rem;
    color: #d1d5db;
    text-transform: capitalize;
    flex-shrink: 0;
}
.bar-track {
    flex-grow: 1;
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 8px;
}
.bar-pct {
    width: 42px;
    text-align: right;
    font-size: 0.82rem;
    color: #9ca3af;
    flex-shrink: 0;
}

.reply-card {
    margin-top: 1.4rem;
    background: rgba(255,255,255,0.05);
    border-left: 4px solid;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    font-size: 1.03rem;
    line-height: 1.55;
    color: #f3f4f6;
}
.reply-label {
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.35rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Emotion styling map (emoji, color)
# ----------------------------
EMOTION_STYLE = {
    "joy":       {"emoji": "😄", "color": "#fbbf24"},
    "happiness": {"emoji": "😄", "color": "#fbbf24"},
    "sadness":   {"emoji": "😢", "color": "#60a5fa"},
    "anger":     {"emoji": "😠", "color": "#f87171"},
    "fear":      {"emoji": "😨", "color": "#a78bfa"},
    "love":      {"emoji": "🥰", "color": "#f472b6"},
    "surprise":  {"emoji": "😲", "color": "#34d399"},
    "neutral":   {"emoji": "😐", "color": "#9ca3af"},
    "disgust":   {"emoji": "🤢", "color": "#84cc16"},
    "shame":     {"emoji": "😳", "color": "#fb923c"},
    "guilt":     {"emoji": "😔", "color": "#94a3b8"},
}
DEFAULT_STYLE = {"emoji": "✨", "color": "#a78bfa"}

def style_for(label):
    return EMOTION_STYLE.get(label.lower(), DEFAULT_STYLE)

# ----------------------------
# Templated replies per emotion
# ----------------------------
REPLIES = {
    "joy": [
        "That's wonderful to hear! Your happiness really comes through — enjoy this moment. 🎉",
        "Love this energy! Whatever's going right for you, savor it.",
        "This made me smile. Keep riding this wave of good feeling!",
    ],
    "happiness": [
        "That's wonderful to hear! Your happiness really comes through — enjoy this moment. 🎉",
        "Love this energy! Whatever's going right for you, savor it.",
    ],
    "sadness": [
        "I'm sorry you're feeling this way. It's okay to sit with it for a moment — you don't have to have it all figured out right now.",
        "That sounds heavy. Be gentle with yourself today.",
        "Thank you for sharing that. Whatever's weighing on you, you don't have to carry it alone.",
    ],
    "anger": [
        "That sounds really frustrating. It's completely valid to feel this way — take a breath before responding to whatever's causing it.",
        "I hear the frustration in that. Sometimes it helps to name exactly what's bothering you most.",
        "That's a lot to be upset about. Your reaction makes sense given what you're describing.",
    ],
    "fear": [
        "That sounds unsettling. Uncertainty is hard — try to focus on what's actually in your control right now.",
        "It's natural to feel anxious about this. You're allowed to take it one step at a time.",
        "That worry makes sense. Would it help to break this down into smaller, less overwhelming pieces?",
    ],
    "love": [
        "That's really sweet. It's clear how much this means to you. 💕",
        "What a warm thing to share — hold onto that feeling.",
    ],
    "surprise": [
        "Whoa, that's unexpected! How are you processing it?",
        "That definitely came out of nowhere — take a moment to let it sink in.",
    ],
    "neutral": [
        "Got it, thanks for sharing that.",
        "Noted — let me know if there's more to it.",
    ],
    "disgust": [
        "That sounds really off-putting. Your reaction is completely understandable.",
        "Yeah, that's a valid thing to be repelled by.",
    ],
    "shame": [
        "That sounds hard to carry. Try not to be too harsh on yourself — everyone stumbles sometimes.",
    ],
    "guilt": [
        "It sounds like this is weighing on you. Acknowledging it is already a good step forward.",
    ],
}
DEFAULT_REPLIES = ["Thanks for sharing that with me."]

def generate_reply(label):
    return random.choice(REPLIES.get(label.lower(), DEFAULT_REPLIES))

# ----------------------------
# Model + tokenizer loading (cached)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    tokenizer_json_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_json_path)
    tokenizer.enable_truncation(max_length=MAX_LEN)
    tokenizer.enable_padding(length=MAX_LEN)

    # id2label: try config.json in tokenizer_files or project root, else fall back
    id2label = None
    for candidate in [os.path.join(TOKENIZER_DIR, "config.json"), "config.json"]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                cfg = json.load(f)
            if "id2label" in cfg:
                id2label = {int(k): v for k, v in cfg["id2label"].items()}
            break

    return session, tokenizer, id2label

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict(text, session, tokenizer, id2label):
    encoding = tokenizer.encode(text)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

    input_names = [i.name for i in session.get_inputs()]
    feed = {}
    if "input_ids" in input_names:
        feed["input_ids"] = input_ids
    if "attention_mask" in input_names:
        feed["attention_mask"] = attention_mask
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros_like(input_ids)

    outputs = session.run(None, feed)
    logits = outputs[0][0]
    probs = softmax(logits)

    num_classes = len(probs)
    if id2label is None:
        id2label = {i: f"label_{i}" for i in range(num_classes)}

    scored = [(id2label.get(i, f"label_{i}"), float(probs[i])) for i in range(num_classes)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# ----------------------------
# UI
# ----------------------------
st.markdown("""
<div class="hero">
    <h1>🌈 EmotionSense</h1>
    <p>Tell me what's on your mind — I'll sense the emotion and respond.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

with st.container():
    text = st.text_area(
        "Your message",
        placeholder="e.g. I just got some really great news and I can't stop smiling...",
        height=120,
        label_visibility="collapsed",
    )
    analyze = st.button("✨ Analyze Emotion")

if analyze:
    if not text or not text.strip():
        st.warning("Please type something first.")
    else:
        try:
            with st.spinner("Reading between the lines..."):
                session, tokenizer, id2label = load_model()
                scored = predict(text, session, tokenizer, id2label)
                time.sleep(0.2)

            top_label, top_score = scored[0]
            top_style = style_for(top_label)

            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-emotion">
                <span class="emoji">{top_style['emoji']}</span>
                <span class="label" style="color:{top_style['color']}">{top_label}</span>
            </div>
            <div class="result-conf">{top_score*100:.1f}% confidence</div>
            """, unsafe_allow_html=True)

            bars_html = ""
            for label, score in scored:
                s = style_for(label)
                bars_html += f"""
                <div class="bar-row">
                    <div class="bar-label">{s['emoji']} {label}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{score*100:.1f}%; background:{s['color']};"></div>
                    </div>
                    <div class="bar-pct">{score*100:.0f}%</div>
                </div>
                """
            st.markdown(bars_html, unsafe_allow_html=True)

            reply = generate_reply(top_label)
            st.markdown(f"""
            <div class="reply-card" style="border-color:{top_style['color']}">
                <div class="reply-label">Suggested reply</div>
                {reply}
            </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong loading or running the model: {e}")
            st.info("Check that model.onnx and tokenizer_files/tokenizer.json are in the project root.")

st.markdown("<br>", unsafe_allow_html=True)