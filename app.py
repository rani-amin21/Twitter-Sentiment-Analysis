import streamlit as st
import pickle
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

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

    if -0.05 < score['compound'] < 0.05:
        return "Neutral", score

    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    result = model.predict(vec)[0]

    if result == 0:
        return "Negative", score
    else:
        return "Positive", score

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["About Project", "Live Analyzer", "Batch Analyzer"])

# -----------------------------
# ABOUT PAGE
# -----------------------------
if page == "About Project":
    st.title("🐦 Twitter Sentiment Analysis")

    st.markdown("""
    ### 🚀 Advanced Sentiment Analysis System

    This app uses:
    - Logistic Regression (ML Model)
    - TF-IDF Vectorization
    - VADER Sentiment Analysis

    ### 💡 Features:
    - Real-time prediction
    - Neutral detection using VADER
    - Visualization dashboard
    - Batch file analysis
    """)

# -----------------------------
# LIVE ANALYZER
# -----------------------------
elif page == "Live Analyzer":

    st.title("🐦 Twitter Sentiment Analyzer")

    user_input = st.text_area("Enter Tweet Text", height=150)

    if st.button("Analyze"):

        if user_input.strip() == "":
            st.warning("Enter some text!")
        else:
            sentiment, score = final_prediction(user_input)

            # RESULT DISPLAY
            st.subheader("🔍 Prediction Result")

            if sentiment == "Positive":
                st.success(f"😊 {sentiment}")
            elif sentiment == "Negative":
                st.error(f"😡 {sentiment}")
            else:
                st.warning(f"😐 {sentiment}")

            # -----------------------------
            # VADER SCORES VISUALIZATION
            # -----------------------------
            st.subheader("📊 Sentiment Scores")

            st.write("Positive:", score['pos'])
            st.progress(score['pos'])

            st.write("Negative:", score['neg'])
            st.progress(score['neg'])

            st.write("Neutral:", score['neu'])
            st.progress(score['neu'])

            # PIE CHART
            fig, ax = plt.subplots()
            labels = ['Positive', 'Negative', 'Neutral']
            values = [score['pos'], score['neg'], score['neu']]

            ax.pie(values, labels=labels, autopct='%1.1f%%')
            st.pyplot(fig)

# -----------------------------
# BATCH ANALYZER (NEW FEATURE)
# -----------------------------
elif page == "Batch Analyzer":

    st.title("📂 Bulk Tweet Analyzer")

    uploaded_file = st.file_uploader("Upload CSV file with 'text' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("CSV must contain 'text' column")
        else:
            results = []

            for text in df["text"]:
                sentiment, _ = final_prediction(str(text))
                results.append(sentiment)

            df["Sentiment"] = results

            st.dataframe(df)

            # BAR CHART
            st.subheader("📊 Sentiment Distribution")
            counts = df["Sentiment"].value_counts()

            st.bar_chart(counts)

            # DOWNLOAD BUTTON
            st.download_button(
                label="Download Results",
                data=df.to_csv(index=False),
                file_name="results.csv",
                mime="text/csv"
            )