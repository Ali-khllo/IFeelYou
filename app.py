import os
import random
import requests
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Make the app take up the full width of the screen for our dashboard layout
st.set_page_config(page_title="IFeelYou", layout="wide")

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

st.title("🤖 IFeelYou — Emotion Detector")
st.write("Enter some text and I'll tell you what emotion it expresses — and suggest how to respond.")

with st.spinner("Loading model..."):
    model, tokenizer = load_model()

# --- Initialize Session State ---
# This ensures our AI results don't vanish when we click other buttons!
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

text = st.text_area("Your text:", value=st.session_state.get("last_text", ""))

if st.button("Analyze"):
    if text.strip():
        st.session_state.last_text = text
        st.toast("Analyzing emotions...", icon="✨")

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        labels = model.config.id2label
        top_idx = probs.argmax().item()
        emotion = labels[top_idx].lower()

        # Save all results to session state
        st.session_state.probs = probs.tolist() # Convert tensor to list for safe storage
        st.session_state.labels = labels
        st.session_state.emotion = emotion
        st.session_state.top_idx = top_idx
        st.session_state.analysis_complete = True

        # Trigger festive animations based on emotion
        if emotion in ["joy", "surprise"]:
            st.balloons()
        elif emotion in ["sadness", "fear"]:
            st.snow()
    else:
        st.warning("Please enter some text first.")

# --- Dashboard UI Layout ---
# Only show the dashboard if we have completed an analysis
if st.session_state.analysis_complete:
    st.divider()
    
    # Create two side-by-side columns
    col1, col2 = st.columns(2)

    # LEFT COLUMN: Human Interaction & Responses
    with col1:
        st.subheader("💬 Dialogue Manager")
        
        predicted_emotion = st.session_state.emotion
        confidence = st.session_state.probs[st.session_state.top_idx] * 100
        
        st.success(f"Primary Emotion: **{predicted_emotion.upper()}** ({confidence:.1f}% confidence)")

        st.write("**Suggested Response:**")
        if predicted_emotion in RESPONSES:
            # We display a random choice, and the button below simply reruns the app to get a new one!
            suggestion = random.choice(RESPONSES[predicted_emotion])
            st.info(suggestion)
            
            # Because we used session_state, clicking this won't erase the screen anymore!
            st.button("🔄 Show another suggestion") 
        else:
            st.write("No template available for this emotion yet.")

    # RIGHT COLUMN: AI Analytics & Math
    with col2:
        st.subheader("🧠 Model Confidence Breakdown")
        
        # Build a Pandas DataFrame to feed into the Streamlit Bar Chart
        chart_data = pd.DataFrame({
            "Emotion": [label.capitalize() for label in st.session_state.labels.values()],
            "Confidence (%)": [p * 100 for p in st.session_state.probs]
        }).set_index("Emotion")
        
        # Display the beautiful interactive bar chart you requested
        st.bar_chart(chart_data, color="#6C63FF")