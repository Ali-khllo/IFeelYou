import os
import random
import requests
import streamlit as st
from streamlit_lottie import st_lottie

# Page setup for cinematic widescreen
st.set_page_config(page_title="IFeelYou — Cinema Mode", page_icon="🎬", layout="wide")

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

# Helper function to load Lottie animations via URL
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

# Cinematic Lottie assets mapping
LOTTIE_URLS = {
    "joy": "https://assets10.lottiefiles.com/packages/lf20_jbrw3hcz.json",
    "sadness": "https://assets9.lottiefiles.com/packages/lf20_96bov9lh.json",
    "anger": "https://assets8.lottiefiles.com/packages/lf20_tlf381fe.json",
    "fear": "https://assets3.lottiefiles.com/packages/lf20_kx22x0w0.json",
    "surprise": "https://assets4.lottiefiles.com/packages/lf20_8eb44f9c.json",
    "disgust": "https://assets5.lottiefiles.com/packages/lf20_9w8wpxie.json",
    "neutral": "https://assets2.lottiefiles.com/packages/lf20_mDnmhAgZkb.json"
}

# Color palettes for cinematic feel
EMOTION_THEMES = {
    "joy": {"color": "#FFD700", "bg": "rgba(255, 215, 0, 0.1)", "glow": "#FFA500"},
    "sadness": {"color": "#4682B4", "bg": "rgba(70, 130, 180, 0.1)", "glow": "#1E90FF"},
    "anger": {"color": "#FF4500", "bg": "rgba(255, 69, 0, 0.15)", "glow": "#FF0000"},
    "fear": {"color": "#9370DB", "bg": "rgba(147, 112, 219, 0.1)", "glow": "#8A2BE2"},
    "surprise": {"color": "#FF007F", "bg": "rgba(255, 0, 127, 0.15)", "glow": "#FF1493"},
    "disgust": {"color": "#3CB371", "bg": "rgba(60, 179, 113, 0.1)", "glow": "#2E8B57"},
    "neutral": {"color": "#E0E0E0", "bg": "rgba(220, 220, 220, 0.05)", "glow": "#A9A9A9"}
}

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

EMOTION_ICONS = {
    "joy": "🌟",
    "sadness": "🌧️",
    "anger": "🔥",
    "fear": "🛡️",
    "surprise": "⚡",
    "disgust": "🤢",
    "neutral": "🎬",
}

# Apply custom dark cinema CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .cinema-card {
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.5s ease-in-out;
    }
    .movie-title {
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ff8c00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='movie-title'>🎬 IFeelYou — Emotional Cinema</h1>", unsafe_allow_html=True)
st.write("<p style='text-align: center; color: #888;'>Script your thoughts. Watch the scene shift in real time.</p>", unsafe_allow_html=True)

with st.spinner("Preparing the cinema projector..."):
    model, tokenizer = load_model()

text = st.text_area("Your Script Line / Thought:", placeholder="Type your story line here...")

def get_random_response(emotion_label):
    options = RESPONSES.get(emotion_label, [])
    if not options:
        return "No template available for this emotion yet."
    current = st.session_state.get("current_suggestion")
    available = [res for res in options if res != current] or options
    return random.choice(available)

if st.button("🎬 Analyze Scene"):
    if text.strip():
        import torch

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        labels = model.config.id2label
        top_idx = probs.argmax().item()
        emotion = labels[top_idx].lower()

        st.session_state["analyzed"] = True
        st.session_state["emotion"] = emotion
        st.session_state["probs"] = probs.tolist()
        st.session_state["labels"] = labels
        st.session_state["top_idx"] = top_idx
        st.session_state["current_suggestion"] = get_random_response(emotion)

        if emotion == "joy":
            st.balloons()
            st.toast("Scene Mood: Pure Joy! 🎉", icon="🎉")
        elif emotion == "sadness":
            st.snow()
            st.toast("Scene Mood: Melancholy... 🌧️", icon="🌧️")
        elif emotion == "anger":
            st.toast("Scene Mood: High Tension 🔥", icon="🔥")
        elif emotion == "fear":
            st.snow()
            st.toast("Scene Mood: Thriller Suspense 🛡️", icon="🛡️")
        elif emotion == "surprise":
            st.balloons()
            st.toast("Scene Mood: Plot Twist! ⚡", icon="⚡")
        else:
            st.toast("Scene Mood: Calm Dialogue ✨", icon="💬")
    else:
        st.warning("Please enter your script line first.")
        st.session_state["analyzed"] = False

# Render Cinema Results
if st.session_state.get("analyzed", False):
    emotion = st.session_state["emotion"]
    probs = st.session_state["probs"]
    labels = st.session_state["labels"]
    top_idx = st.session_state["top_idx"]
    theme = EMOTION_THEMES.get(emotion, EMOTION_THEMES["neutral"])
    icon = EMOTION_ICONS.get(emotion, "🎬")

    # Layout into 2 columns: Left = Lottie Movie Scene, Right = Suggested Response
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown(f"### {icon} Scene Mood: **{emotion.upper()}**")
        lottie_json = load_lottieurl(LOTTIE_URLS.get(emotion, LOTTIE_URLS["neutral"]))
        if lottie_json:
            st_lottie(lottie_json, height=220, key=f"lottie_{emotion}")
        else:
            st.write(f"*(Animation active for {emotion})*")

    with col2:
        # Dynamic theme container
        st.markdown(
            f"""
            <div class="cinema-card" style="background: {theme['bg']}; border: 2px solid {theme['color']};">
                <h3 style="color: {theme['color']}; margin-top: 0;">💬 Director's Response Line</h3>
                <p style="font-size: 1.25rem; line-height: 1.5; font-weight: 500;">"{st.session_state['current_suggestion']}"</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🔄 Take 2 (Show Another Response)"):
            st.session_state["current_suggestion"] = get_random_response(emotion)
            st.rerun()

    st.divider()

    # --- TOP 3 SCORES WITH GLOWING THEME BARS ---
    st.write("### 🍿 Top 3 Scene Scores")
    scores = [(labels[str(i) if isinstance(labels, dict) and str(i) in labels else i], probs[i]) for i in range(len(probs))]
    top_3_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:3]

    for label, score in top_3_scores:
        lbl_lower = label.lower()
        lbl_icon = EMOTION_ICONS.get(lbl_lower, "📊")
        lbl_theme = EMOTION_THEMES.get(lbl_lower, EMOTION_THEMES["neutral"])

        col_a, col_b = st.columns([1, 4])
        with col_a:
            st.markdown(f"<span style='color: {lbl_theme['color']}; font-weight: bold;'>{lbl_icon} {label.capitalize()}</span>", unsafe_allow_html=True)
            st.write(f"**{score * 100:.1f}%**")
        with col_b:
            st.progress(score)