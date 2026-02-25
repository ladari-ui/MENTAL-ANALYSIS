import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os
import matplotlib.pyplot as plt
import joblib
import numpy as np

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Mental Health Emotion Analyzer", layout="wide")

# -------------------------
# LOAD TRAINED MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------
# PREDICTION FUNCTION (ML BASED)
# -------------------------
def detect_emotion(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    confidence = np.max(probabilities) * 100
    return prediction, confidence

# -------------------------
# CUSTOM STYLING
# -------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f4f8fb;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# SIDEBAR NAVIGATION
# -------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Intro", "📝 Input", "📊 Output", "💬 Feedback"])

# -------------------------
# INTRO PAGE
# -------------------------
if page == "🏠 Intro":

    st.title("🧠 Mental Health Emotion Analyzer")
    st.markdown("### AI-Based Emotion Detection System 💙")

    st.write("""
    This application:
    - Uses TF-IDF Vectorization
    - Uses Logistic Regression
    - Predicts real emotions using trained ML model
    - Accepts Text and Voice input
    - Shows prediction confidence
    """)

    st.success("👈 Go to Input page to start!")

# -------------------------
# INPUT PAGE
# -------------------------
elif page == "📝 Input":

    st.title("📝 Enter Your Feelings")

    # TEXT INPUT
    st.subheader("✍ Text Input")
    user_text = st.text_area("Type how you feel...")

    if st.button("Analyze Text"):
        if user_text.strip() != "":
            emotion, confidence = detect_emotion(user_text)

            st.session_state["emotion"] = emotion
            st.session_state["confidence"] = confidence
            st.session_state["text"] = user_text

            st.success("Emotion analyzed! Go to Output page 📊")
        else:
            st.warning("Please enter text.")

    st.markdown("---")

    # VOICE INPUT
    st.subheader("🎤 Voice Input")
    audio_bytes = st.audio_input("Start Recording")

    if audio_bytes is not None:
        try:
            with st.spinner("Processing audio..."):

                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
                    temp_audio.write(audio_bytes.read())
                    webm_path = temp_audio.name

                wav_path = webm_path.replace(".webm", ".wav")
                sound = AudioSegment.from_file(webm_path)
                sound.export(wav_path, format="wav")

                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)

                emotion, confidence = detect_emotion(text)

                st.session_state["emotion"] = emotion
                st.session_state["confidence"] = confidence
                st.session_state["text"] = text

                st.success("Voice analyzed! Go to Output page 📊")

                os.remove(webm_path)
                os.remove(wav_path)

        except sr.UnknownValueError:
            st.error("Could not recognize audio. Speak clearly.")
        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------
# OUTPUT PAGE
# -------------------------
elif page == "📊 Output":

    st.title("📊 Emotion Analysis Result")

    if "emotion" in st.session_state:

        st.write("### 🗣 Input Text")
        st.info(st.session_state["text"])

        st.write("### 🎯 Predicted Emotion")
        st.success(st.session_state["emotion"])

        st.write(f"### 📈 Confidence: {st.session_state['confidence']:.2f}%")

        # Visualization
        emotion_classes = model.classes_
        text_vec = vectorizer.transform([st.session_state["text"]])
        probabilities = model.predict_proba(text_vec)[0]

        fig, ax = plt.subplots()
        ax.bar(emotion_classes, probabilities)
        ax.set_ylabel("Probability")
        ax.set_title("Emotion Probability Distribution")
        plt.xticks(rotation=45)

        st.pyplot(fig)

    else:
        st.warning("Please analyze emotion first from Input page.")

# -------------------------
# FEEDBACK PAGE
# -------------------------
elif page == "💬 Feedback":

    st.title("💬 Feedback")

    rating = st.slider("Rate this app (1-5)", 1, 5)
    feedback = st.text_area("Write your feedback...")

    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! 💙")
        st.write("⭐ Rating:", rating)
        st.write("📝 Feedback:", feedback)