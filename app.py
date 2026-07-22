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

ILHAM_QUOTES = {
    "joy": [
        "Pak Ilham says: 'A+ energy right here! Keep that positive momentum going!' 🌟",
        "Pak Ilham says: 'This code runs with 0 bugs and 100% good vibes!' 🚀",
        "Pak Ilham says: 'Seeing this level of enthusiasm makes teaching totally worth it!' 😊"
    ],
    "sadness": [
        "Pak Ilham says: 'Even the best algorithms hit edge cases. Take a warm drink break, you're doing great!' ☕",
        "Pak Ilham says: 'Debugging life takes patience. Be kind to yourself today!' 💙",
        "Pak Ilham says: 'Don't worry, every error log is just a stepping stone to a clean build.' 🌱"
    ],
    "anger": [
        "Pak Ilham says: 'Deep breaths! Step away from the keyboard for 5 minutes — solutions pop up when you relax!' 🧘‍♂️",
        "Pak Ilham says: 'Frustration is curiosity in disguise. Let's break the problem down into smaller functions!' 💡",
        "Pak Ilham says: 'Channel that fiery energy into writing an incredible piece of code!' 🔥"
    ],
    "fear": [
        "Pak Ilham says: 'Uncertainty is just an unexplored branch in your logic tree. You've got this!' 🛡️",
        "Pak Ilham says: 'No stress! Even senior devs consult docs daily. Step by step!' 📚",
        "Pak Ilham says: 'Take a slow breath. You are much more capable than you give yourself credit for!' ✨"
    ],
    "surprise": [
        "Pak Ilham says: 'Plot twist! That's the beauty of unpredictable data — full of surprises!' ⚡",
        "Pak Ilham says: 'An unexpected output! Let's inspect the variables and enjoy the discovery!' 🔍"
    ],
    "disgust": [
        "Pak Ilham says: 'Ugh, tough situations are rough! Let's refactor and start fresh!' 🧹",
        "Pak Ilham says: 'Totally fair reaction. Shake it off, take a breath, and let's clear the queue!' ✨"
    ],
    "neutral": [
        "Pak Ilham says: 'Smooth, steady, and balanced — ready for whatever prompt comes next!' 🎯",
        "Pak Ilham says: 'Clear and concise! Perfect baseline for further exploration.' 💻"
    ]
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
    .mobile-header {
        text-align: center;
        padding: 10px;
        border-radius: 16px;
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        color: white;
        margin-bottom: 20px;
    }
    .ilham-card {
        padding: 16px;
        border-radius: 16px;
        background-color: #F3F4F6;
        border-left: 6px solid #6C63FF;
        margin-top: 15px;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="mobile-header">
    <h2 style="margin:0; color: white;">📱 IFeelYou</h2>
    <p style="margin:0; font-size: 0.9rem; opacity: 0.9;">Emotion Detector & Assistant</p>
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

        # TRIGGER UNIQUE ANIMATION FOR EVERY EMOTION
        if emotion == "joy":
            st.balloons()
            st.toast("Celebrating your joy! 🎉", icon="🎉")
        elif emotion == "sadness":
            st.snow()
            st.toast("Sending warmth your way... 🌧️", icon="🌧️")
        elif emotion == "anger":
            st.toast("Deep breath... releasing the tension 🔥", icon="🔥")
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

# --- MOBILE RESPONSIVE DISPLAY FLOW ---
if st.session_state.analysis_complete:
    st.divider()
    
    predicted_emotion = st.session_state.emotion
    confidence = st.session_state.probs[st.session_state.top_idx] * 100
    theme = EMOTION_THEMES.get(predicted_emotion, EMOTION_THEMES["neutral"])

    # 1. Primary Result Banner
    st.markdown(
        f"""
        <div style="background-color: {theme['bg']}; padding: 15px; border-radius: 12px; border: 2px solid {theme['color']}; text-align: center;">
            <h3 style="color: {theme['color']}; margin: 0;">{theme['icon']} {predicted_emotion.upper()} ({confidence:.1f}%)</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Suggested Response Section
    st.markdown("### 💬 Suggested Response")
    if predicted_emotion in RESPONSES:
        suggestion = random.choice(RESPONSES[predicted_emotion])
        st.info(suggestion)
        st.button("🔄 Another Suggestion", use_container_width=True)
    else:
        st.write("No template available for this emotion yet.")

    # 3. Pak Ilham's Cute Quote Card
    ilham_quote = random.choice(ILHAM_QUOTES.get(predicted_emotion, ["Pak Ilham says: 'Keep up the great work!' ✨"]))
    st.markdown(
        f"""
        <div class="ilham-card">
            <strong>🎓 Pak Ilham's Take:</strong><br>
            "{ilham_quote.replace('Pak Ilham says: ', '')}"
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4. Top 3 Confidence Chart
    st.markdown("---")
    st.markdown("### 📊 Top 3 Detected Emotions")
    
    scores = [(st.session_state.labels[str(i) if isinstance(st.session_state.labels, dict) and str(i) in st.session_state.labels else i], st.session_state.probs[i]) for i in range(len(st.session_state.probs))]
    top_3_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:3]

    for label, score in top_3_scores:
        lbl_lower = label.lower()
        lbl_theme = EMOTION_THEMES.get(lbl_lower, EMOTION_THEMES["neutral"])
        
        st.write(f"**{lbl_theme['icon']} {label.capitalize()}**: {score * 100:.1f}%")
        st.progress(score)