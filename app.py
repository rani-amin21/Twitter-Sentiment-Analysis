import streamlit as st
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")

# -----------------------------
# CUSTOM CSS (Twitter Theme)
# -----------------------------
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }

    .main {
        background-color: #0e1117;
    }

    h1, h2, h3 {
        color: #1DA1F2;
        text-align: center;
    }

    .stButton>button {
        background-color: #1DA1F2;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }

    .stTextArea textarea {
        background-color: #192734;
        color: white;
        border-radius: 10px;
    }

    </style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL & VECTORIZER
# -----------------------------
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

analyzer = SentimentIntensityAnalyzer()


# -----------------------------
# PREPROCESS FUNCTION (same as notebook)
# -----------------------------
def preprocess_text(text):
    import re
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


# -----------------------------
# FINAL PREDICTION FUNCTION
# -----------------------------
def final_prediction(text):
    # VADER for Neutral
    score = analyzer.polarity_scores(text)
    if -0.05 < score['compound'] < 0.05:
        return "Neutral"

    # ML Model
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    result = model.predict(vec)[0]

    if result == 0:
        return "Negative"
    else:
        return "Positive"


# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About Project", "Live Analyzer"])

# -----------------------------
# ABOUT PAGE
# -----------------------------
if page == "About Project":
    st.title("Twitter Sentiment Analysis")

    st.markdown("""
    ### 📌 Project Description

    This project analyzes sentiment of Twitter text using:

    - Logistic Regression (Machine Learning)
    - TF-IDF Vectorization
    - VADER (for Neutral sentiment)

    ### 🎯 Features
    - Predicts Positive, Negative, Neutral sentiment
    - Real-time text analysis
    - Hybrid ML + NLP approach

    ### 🛠️ Technologies Used
    - Python
    - Scikit-learn
    - Streamlit
    - NLP (VADER)
    """)

# -----------------------------
# LIVE ANALYZER PAGE
# -----------------------------
elif page == "Live Analyzer":

    st.markdown("<h1>Twitter Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Enter text to analyze sentiment</h4>", unsafe_allow_html=True)

    st.write("")

    # Text Input Box (like your screenshot)
    user_input = st.text_area("Type or paste text to analyze sentiment...", height=200)

    st.write("")

    # Analyze Button
    if st.button("🚀 Analyze"):

        if user_input.strip() == "":
            st.warning("Please enter some text")
        else:
            result = final_prediction(user_input)

            # Output Styling
            if result == "Positive":
                st.success(f"😊 Sentiment: {result}")
            elif result == "Negative":
                st.error(f"😡 Sentiment: {result}")
            else:
                st.warning(f"😐 Sentiment: {result}")