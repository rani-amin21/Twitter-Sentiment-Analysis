import streamlit as st
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
analyzer = SentimentIntensityAnalyzer()

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess_text(text):
    import re
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# -----------------------------
# FINAL PREDICTION
# -----------------------------
def final_prediction(text):
    score = analyzer.polarity_scores(text)

    if score['compound'] >= 0.05:
        return "Positive", score
    elif score['compound'] <= -0.05:
        return "Negative", score
    else:
        return "Neutral", score

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["About Project", "Live Analyzer"])

# -----------------------------
# ABOUT PAGE
# -----------------------------
if page == "About Project":
    st.title("Twitter Sentiment Analysis")

    st.markdown("""
    ### Project Description

    This project analyzes sentiment of Twitter text using:

    - Logistic Regression (Machine Learning)
    - TF-IDF Vectorization
    - VADER (for accurate sentiment detection)

    ### Features
    - Predicts Positive, Negative, Neutral sentiment
    - Real-time text analysis
    - Hybrid ML + NLP approach

    ### Technologies Used
    - Python
    - Scikit-learn
    - Streamlit
    - NLP (VADER)
    """)

# -----------------------------
# LIVE ANALYZER
# -----------------------------
elif page == "Live Analyzer":

    st.title("Twitter Sentiment Analyzer")

    user_input = st.text_area("Enter Tweet Text", height=150)

    if st.button("Analyze"):

        if user_input.strip() == "":
            st.warning("Please enter some text")
        else:
            sentiment, score = final_prediction(user_input)

            # RESULT
            st.subheader("🔍 Prediction Result")

            if sentiment == "Positive":
                st.success(f"😊 {sentiment}")
            elif sentiment == "Negative":
                st.error(f"😡 {sentiment}")
            else:
                st.warning(f"😐 {sentiment}")

            # PERCENTAGE
            pos = int(score['pos'] * 100)
            neg = int(score['neg'] * 100)
            neu = int(score['neu'] * 100)

            # SCORES
            st.subheader("📊 Sentiment Scores")

            st.write(f"Positive: {pos}%")
            st.progress(score['pos'])

            st.write(f"Negative: {neg}%")
            st.progress(score['neg'])

            st.write(f"Neutral: {neu}%")
            st.progress(score['neu'])