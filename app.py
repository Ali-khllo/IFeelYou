import os
import random
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
    "joy": "🎉",
    "sadness": "🌧️",
    "anger": "🔥",
    "fear": "🛡️",
    "surprise": "⚡",
    "disgust": "🤢",
    "neutral": "💬",
}

st.title("IFeelYou — Emotion Detector")
st.write("Enter some text and I'll tell you what emotion it expresses — and suggest how to respond.")

with st.spinner("Loading model..."):
    model, tokenizer = load_model()

text = st.text_area("Your text:")

# Helper function to pick a response without repeating the exact same one immediately
def get_random_response(emotion_label):
    options = RESPONSES.get(emotion_label, [])
    if not options:
        return "No template available for this emotion yet."
    current = st.session_state.get("current_suggestion")
    available = [res for res in options if res != current] or options
    return random.choice(available)

if st.button("Analyze"):
    if text.strip():
        import torch

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        labels = model.config.id2label
        top_idx = probs.argmax().item()
        emotion = labels[top_idx].lower()

        # Save results in session state so re-runs (like clicking refresh) preserve state
        st.session_state["analyzed"] = True
        st.session_state["emotion"] = emotion
        st.session_state["probs"] = probs.tolist()
        st.session_state["labels"] = labels
        st.session_state["top_idx"] = top_idx
        st.session_state["current_suggestion"] = get_random_response(emotion)

        # Trigger dynamic animations
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
            st.toast("Ugh, totally understandable reaction 🤢", icon="🤢")
        else:
            st.toast("Got it, steady and balanced ✨", icon="💬")
    else:
        st.warning("Please enter some text first.")
        st.session_state["analyzed"] = False

# Render results whenever an analysis exists in state
if st.session_state.get("analyzed", False):
    emotion = st.session_state["emotion"]
    probs = st.session_state["probs"]
    labels = st.session_state["labels"]
    top_idx = st.session_state["top_idx"]
    icon = EMOTION_ICONS.get(emotion, "💬")

    # --- 1. SUGGESTED RESPONSE ON TOP ---
    st.subheader(f"{icon} Suggested Response")
    st.info(st.session_state["current_suggestion"])

    # Refresh button picks a new string from the SAME emotion list
    if st.button("🔄 Show another suggestion"):
        st.session_state["current_suggestion"] = get_random_response(emotion)
        st.rerun()

    st.divider()

    # --- 2. PREDICTED EMOTION & CONFIDENCE ---
    st.success(f"Predicted emotion: **{emotion.upper()}** {icon} ({probs[top_idx]*100:.1f}% confidence)")

    # --- 3. TOP 3 SCORES ONLY ---
    st.write("### Top 3 Detected Emotions:")
    scores = [(labels[str(i) if isinstance(labels, dict) and str(i) in labels else i], probs[i]) for i in range(len(probs))]
    top_3_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:3]

    for label, score in top_3_scores:
        lbl_lower = label.lower()
        lbl_icon = EMOTION_ICONS.get(lbl_lower, "📊")
        st.write(f"**{lbl_icon} {label.capitalize()}**: {score * 100:.1f}%")
        st.progress(score)