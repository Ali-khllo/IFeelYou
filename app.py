import os
import random
import requests
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configured for mobile viewport
st.set_page_config(page_title="IFeelYou — Mobile Edition", layout="centered", page_icon="📱")

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

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

RESPONSES = {
    "anger": [
        "I hear you, and it's okay to feel this frustrated.",
        "That sounds really infuriating — I'd be upset too.",
        "Let's take a breath together before responding to this.",
        "Your anger makes sense given what happened.",
        "I'm here if you want to vent more about this.",
        "It's valid to be mad right now, take your time.",
        "I understand why this is making you so angry.",
        "Would it help to talk through what triggered this?",
        "I'm on your side — this situation sounds unfair.",
        "Let's figure out a way to make this better together.",
    ],
    "disgust": [
        "Yeah, that's genuinely gross — I get the reaction.",
        "I understand why that would turn your stomach.",
        "That's an unpleasant thing to have to deal with.",
        "I don't blame you for feeling repulsed by that.",
        "That sounds like something nobody should have to see.",
        "I can see why that left a bad taste in your mouth.",
        "That reaction makes complete sense to me.",
        "Ugh, I'd feel the same way about that.",
        "That's rough — no wonder you're put off.",
        "I hear you, that does sound unpleasant.",
    ],
    "fear": [
        "It's okay to feel scared — you're not alone in this.",
        "That sounds really frightening, I'm here with you.",
        "Take a deep breath, we can work through this together.",
        "Your fear makes sense given the situation.",
        "I understand why that would feel unsettling.",
        "You're safe right now, let's think this through calmly.",
        "It's completely normal to feel anxious about this.",
        "I'm here to help you feel more at ease.",
        "Let's break this down so it feels less overwhelming.",
        "You don't have to face this alone.",
    ],
    "joy": [
        "That's wonderful news, I'm so happy for you!",
        "Your excitement is contagious — congratulations!",
        "This made my day, thank you for sharing it.",
        "That's amazing, you deserve this happiness.",
        "I love seeing you this thrilled about something.",
        "What a great moment — enjoy it fully!",
        "This is fantastic, I'm celebrating with you.",
        "You should be proud of this moment.",
        "That's such uplifting news, thanks for telling me.",
        "I'm grinning just reading this — awesome!",
    ],
    "sadness": [
        "I'm really sorry you're going through this.",
        "It's okay to feel sad, I'm here for you.",
        "That sounds really hard, take all the time you need.",
        "I wish I could make this easier for you.",
        "Your feelings are completely valid right now.",
        "I'm sending you support during this tough time.",
        "It's alright to not be okay today.",
        "I'm listening, and I care about how you feel.",
        "This sounds painful, and I'm here with you.",
        "Take it one step at a time, I've got you.",
    ],
    "surprise": [
        "Wow, I did not see that coming either!",
        "That's quite unexpected — how are you feeling about it?",
        "What a twist! Tell me more about what happened.",
        "That must have caught you completely off guard.",
        "I'm surprised too, that's a lot to take in.",
        "No way — that's such a shock!",
        "That's a plot twist for sure, how do you feel?",
        "I wasn't expecting that at all, wow.",
        "That's wild, what happened next?",
        "Quite the surprise! I'd love to hear more.",
    ],
    "neutral": [
        "Got it, thanks for letting me know.",
        "Understood, is there anything else on your mind?",
        "Okay, noted — how can I help further?",
        "Thanks for sharing that with me.",
        "I see, let me know if you need anything.",
        "Alright, that makes sense.",
        "Noted — what's next on your mind?",
        "Thanks for the update.",
        "I understand, feel free to share more.",
        "Okay, I'm here if you need anything else.",
    ],
}

EMOTION_THEMES = {
    "joy": {"color": "#FFD700", "bg": "#FFFBEB", "icon": "🎉", "bar": "#FFB800"},
    "sadness": {"color": "#1E90FF", "bg": "#F0F8FF", "icon": "🌧️", "bar": "#1E90FF"},
    "anger": {"color": "#FF4500", "bg": "#FFF5F5", "icon": "🔥", "bar": "#FF4500"},
    "fear": {"color": "#8A2BE2", "bg": "#F8F5FF", "icon": "🛡️", "bar": "#8A2BE2"},
    "surprise": {"color": "#FF007F", "bg": "#FFF0F6", "icon": "⚡", "bar": "#FF007F"},
    "disgust": {"color": "#2E8B57", "bg": "#F0FFF4", "icon": "🤢", "bar": "#2E8B57"},
    "neutral": {"color": "#6C63FF", "bg": "#F8F9FA", "icon": "💬", "bar": "#6C63FF"}
}

# Custom Mobile Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;700;800&family=Nunito:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 15% 10%, #FFE3EE 0%, transparent 45%),
                    radial-gradient(circle at 85% 15%, #E4E7FF 0%, transparent 40%),
                    radial-gradient(circle at 20% 90%, #E1FFF0 0%, transparent 40%),
                    #FBF9FF;
    }

    /* floating background blobs */
    .blob {
        position: fixed;
        border-radius: 50%;
        filter: blur(2px);
        opacity: 0.35;
        z-index: 0;
        pointer-events: none;
        animation: drift 14s ease-in-out infinite;
    }
    .blob-a { width: 90px; height: 90px; top: 8%; left: 6%; background: #FFD1E8; animation-delay: 0s; }
    .blob-b { width: 60px; height: 60px; top: 70%; left: 85%; background: #C9D6FF; animation-delay: 2s; }
    .blob-c { width: 45px; height: 45px; top: 40%; left: 90%; background: #C6FFE0; animation-delay: 4s; }

    @keyframes drift {
        0%, 100% { transform: translateY(0px) translateX(0px) scale(1); }
        50% { transform: translateY(-18px) translateX(10px) scale(1.08); }
    }

    @keyframes bounceIn {
        0% { transform: scale(0.6); opacity: 0; }
        60% { transform: scale(1.05); opacity: 1; }
        100% { transform: scale(1); }
    }

    @keyframes popIn {
        0% { transform: translateY(14px) scale(0.95); opacity: 0; }
        100% { transform: translateY(0) scale(1); opacity: 1; }
    }

    @keyframes wiggle {
        0%, 100% { transform: rotate(-6deg); }
        50% { transform: rotate(6deg); }
    }

    @keyframes shimmer {
        0% { background-position: -120px 0; }
        100% { background-position: 220px 0; }
    }

    @keyframes fillBar {
        from { width: 0%; }
    }

    .mobile-header {
        text-align: center;
        padding: 18px 12px;
        border-radius: 22px;
        background: linear-gradient(135deg, #8A7DFF, #FF8FB1);
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 24px rgba(138, 125, 255, 0.35);
        animation: bounceIn 0.6s ease;
        position: relative;
        z-index: 1;
    }
    .mobile-header h2 {
        font-family: 'Baloo 2', sans-serif;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .header-emoji {
        display: inline-block;
        animation: wiggle 1.8s ease-in-out infinite;
    }

    /* cute pill-shaped animated button */
    .stButton > button {
        border-radius: 999px !important;
        font-family: 'Baloo 2', sans-serif !important;
        font-weight: 700 !important;
        border: none !important;
        background: linear-gradient(135deg, #8A7DFF, #FF8FB1) !important;
        color: white !important;
        box-shadow: 0 6px 16px rgba(138, 125, 255, 0.4) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton > button:hover {
        transform: scale(1.04) translateY(-2px) !important;
        box-shadow: 0 10px 22px rgba(255, 143, 177, 0.45) !important;
    }
    .stButton > button:active {
        transform: scale(0.97) !important;
    }

    .result-card {
        padding: 18px;
        border-radius: 20px;
        text-align: center;
        animation: popIn 0.45s ease;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        position: relative;
        z-index: 1;
    }
    .result-card h3 {
        font-family: 'Baloo 2', sans-serif;
        font-weight: 700;
    }
    .result-icon {
        display: inline-block;
        animation: wiggle 1.4s ease-in-out infinite;
    }

    .suggestion-card {
        padding: 14px 16px;
        border-radius: 16px;
        background: #F5F3FF;
        border: 2px dashed #C7BFFF;
        font-size: 0.98rem;
        animation: popIn 0.5s ease;
        margin-bottom: 10px;
    }

    .mini-row {
        border-radius: 14px;
        padding: 10px 12px 6px 12px;
        margin-bottom: 10px;
        background: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
        animation: popIn 0.45s ease both;
    }
    .mini-row .mini-icon { animation: wiggle 2.2s ease-in-out infinite; display: inline-block; }

    .bar-track {
        width: 100%;
        height: 12px;
        border-radius: 999px;
        background: #F0F0F5;
        overflow: hidden;
        margin-top: 4px;
    }
    .bar-fill {
        height: 100%;
        border-radius: 999px;
        background-image: linear-gradient(90deg, rgba(255,255,255,0) 0, rgba(255,255,255,0.6) 50%, rgba(255,255,255,0) 100%);
        background-size: 60px 100%;
        background-repeat: no-repeat;
        animation: fillBar 0.9s ease forwards, shimmer 1.6s linear infinite 0.9s;
    }
</style>

<div class="blob blob-a"></div>
<div class="blob blob-b"></div>
<div class="blob blob-c"></div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="mobile-header">
    <h2 style="margin:0; color: white;"><span class="header-emoji">📱</span> IFeelYou</h2>
    <p style="margin:0; font-size: 0.9rem; opacity: 0.9;">Emotion Detector &amp; Assistant</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading AI model..."):
    model, tokenizer = load_model()

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

text = st.text_area("Your thought or message:", value=st.session_state.get("last_text", ""), placeholder="Type here...", height=100)

if st.button("✨ Analyze Emotion", use_container_width=True):
    if text.strip():
        st.session_state.last_text = text

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        labels = model.config.id2label
        top_idx = probs.argmax().item()
        emotion = labels[top_idx].lower()

        st.session_state.probs = probs.tolist()
        st.session_state.labels = labels
        st.session_state.emotion = emotion
        st.session_state.top_idx = top_idx
        st.session_state.analysis_complete = True

        # TRIGGER UNIQUE ANIMATIONS PER EMOTION
        if emotion == "joy":
            st.balloons()
            st.toast("Celebrating your joy! 🎉", icon="🎉")
        elif emotion == "sadness":
            st.snow()
            st.toast("Sending warmth your way... 🌧️", icon="🌧️")
        elif emotion == "anger":
            st.toast("Deep breath... releasing tension 🔥", icon="🔥")
        elif emotion == "fear":
            st.snow()
            st.toast("You are safe and supported 🛡️", icon="🛡️")
        elif emotion == "surprise":
            st.balloons()
            st.toast("Whoa! What a twist! ⚡", icon="⚡")
        elif emotion == "disgust":
            st.toast("Ugh, totally valid reaction! 🤢", icon="🤢")
        else:
            st.toast("Steady and balanced ✨", icon="💬")
    else:
        st.warning("Please enter some text first.")

# --- MOBILE DISPLAY FLOW ---
if st.session_state.analysis_complete:
    st.divider()
    
    predicted_emotion = st.session_state.emotion
    confidence = st.session_state.probs[st.session_state.top_idx] * 100
    theme = EMOTION_THEMES.get(predicted_emotion, EMOTION_THEMES["neutral"])

    # 1. Primary Result Banner
    st.markdown(
        f"""
        <div class="result-card" style="background-color: {theme['bg']}; border: 3px solid {theme['color']};">
            <h3 style="color: {theme['color']}; margin: 0;">
                <span class="result-icon">{theme['icon']}</span> {predicted_emotion.upper()} &nbsp;·&nbsp; {confidence:.1f}%
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # 2. Suggested Response Section
    st.markdown("### 💬 Suggested Response")
    if predicted_emotion in RESPONSES:
        suggestion = random.choice(RESPONSES[predicted_emotion])
        st.markdown(f'<div class="suggestion-card">🌟 {suggestion}</div>', unsafe_allow_html=True)
        st.button("🔄 Show Another Suggestion", use_container_width=True)
    else:
        st.write("No template available for this emotion yet.")

    # 3. Top 3 Confidence Chart
    st.markdown("---")
    st.markdown("### 📊 Top 3 Detected Emotions")
    
    scores = [(st.session_state.labels[str(i) if isinstance(st.session_state.labels, dict) and str(i) in st.session_state.labels else i], st.session_state.probs[i]) for i in range(len(st.session_state.probs))]
    top_3_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:3]

    for i, (label, score) in enumerate(top_3_scores):
        lbl_lower = label.lower()
        lbl_theme = EMOTION_THEMES.get(lbl_lower, EMOTION_THEMES["neutral"])
        pct = score * 100

        st.markdown(
            f"""
            <div class="mini-row" style="animation-delay: {i * 0.12:.2f}s;">
                <div style="display:flex; justify-content:space-between; font-weight:700; font-family:'Baloo 2',sans-serif;">
                    <span><span class="mini-icon">{lbl_theme['icon']}</span> {label.capitalize()}</span>
                    <span style="color:{lbl_theme['bar']};">{pct:.1f}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct:.1f}%; background-color:{lbl_theme['bar']}; animation-delay: {i * 0.12:.2f}s, {0.9 + i * 0.12:.2f}s;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )