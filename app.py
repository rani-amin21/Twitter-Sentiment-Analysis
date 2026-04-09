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
page = st.sidebar.radio("Go to", ["About Project", "Live Analyzer", "Bulk Analyzer", "Insights Dashboard"])

# -----------------------------
# ABOUT PAGE
# -----------------------------
if page == "About Project":
    st.title("Twitter Sentiment Analysis")

    st.markdown("""
    ### Project Description

    This project analyzes sentiment of Twitter text using:
    - Logistic Regression
    - TF-IDF Vectorization
    - VADER Sentiment Analysis

    ### Features
    - Single tweet analysis
    - Bulk tweet analysis
    - Insights dashboard
    - Word cloud visualization
    
    ### Technologies Used 
    - Python 
    - Scikit-learn 
    - Streamlit 
    - NLP 
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

            st.subheader("🔍 Prediction Result")

            if sentiment == "Positive":
                st.success(f"😊 {sentiment}")
            elif sentiment == "Negative":
                st.error(f"😡 {sentiment}")
            else:
                st.warning(f"😐 {sentiment}")

            import pandas as pd

            df_single = pd.DataFrame({
                "Tweet": [user_input],
                "Sentiment": [sentiment]
            })

            # If already data exists → append
            if "bulk_data" in st.session_state:
                st.session_state["bulk_data"] = pd.concat(
                    [st.session_state["bulk_data"], df_single],
                    ignore_index=True
                )
            else:
                st.session_state["bulk_data"] = df_single

            st.info("Dashboard updated with this tweet")

            overall = int(score['compound'] * 100)

            st.subheader("⭐ Overall Sentiment Score")
            st.write(f"Overall Score: {overall}%")

            slider_value = (score['compound'] + 1) / 2
            st.progress(slider_value)

            confidence = int(abs(score['compound']) * 100)
            st.write(f"Confidence: {confidence}%")

            pos = int(score['pos'] * 100)
            neg = int(score['neg'] * 100)
            neu = int(score['neu'] * 100)

            st.subheader("📊 Sentiment Scores Breakdown")

            st.write(f"Positive: {pos}%")
            st.progress(score['pos'])

            st.write(f"Negative: {neg}%")
            st.progress(score['neg'])

            st.write(f"Neutral: {neu}%")
            st.progress(score['neu'])

# -----------------------------
# BULK ANALYZER
# -----------------------------
elif page == "Bulk Analyzer":

    st.title("📂 Bulk Tweet Analyzer")

    # Manual Input
    st.subheader("✍️ Enter Tweets Manually")
    user_text = st.text_area(
        "Enter tweets (one per line)",
        height=150
    )

    # CSV Upload
    st.subheader("📤 Upload CSV")
    uploaded_file = st.file_uploader("Upload CSV (with 'text' column)", type=["csv"])

    if st.button("🚀 Analyze Tweets"):

        import pandas as pd

        tweets = []

        if user_text.strip() != "":
            tweets = user_text.split("\n")

        elif uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            if "text" not in df.columns:
                st.error("CSV must contain 'text' column")
                st.stop()

            tweets = df["text"].astype(str).tolist()

        else:
            st.warning("Enter tweets or upload CSV")
            st.stop()

        results = []

        for tweet in tweets:
            sentiment, _ = final_prediction(tweet)
            results.append(sentiment)

        df_result = pd.DataFrame({
            "Tweet": tweets,
            "Sentiment": results
        })

        st.subheader("📊 Results")
        st.dataframe(df_result)

        # Overall bulk sentiment
        counts = df_result["Sentiment"].value_counts()

        pos = counts.get("Positive", 0)
        neg = counts.get("Negative", 0)
        neu = counts.get("Neutral", 0)

        st.subheader("⭐ Overall Sentiment of All Tweets")

        if pos > neg and pos > neu:
            st.success("Overall Sentiment: Positive 😊")
        elif neg > pos and neg > neu:
            st.error("Overall Sentiment: Negative 😡")
        else:
            st.warning("Overall Sentiment: Neutral 😐")

        # Show counts also (good for understanding)
        st.write(f"Positive Tweets: {pos}")
        st.write(f"Negative Tweets: {neg}")
        st.write(f"Neutral Tweets: {neu}")

        # Save for dashboard
        st.session_state["bulk_data"] = df_result

# -----------------------------
# INSIGHTS DASHBOARD
# -----------------------------
elif page == "Insights Dashboard":

    st.title("📊 Sentiment Insights Dashboard")

    if "bulk_data" not in st.session_state:
        st.warning("Please analyze tweets first in Bulk Analyzer")
    else:
        df = st.session_state["bulk_data"]

        counts = df["Sentiment"].value_counts()

        st.subheader("📈 Sentiment Distribution")
        st.bar_chart(counts)

        total = len(df)
        pos = counts.get("Positive", 0)
        neg = counts.get("Negative", 0)
        neu = counts.get("Neutral", 0)

        st.subheader("📊 Summary")

        st.write(f"Positive: {pos} ({int(pos/total*100)}%)")
        st.write(f"Negative: {neg} ({int(neg/total*100)}%)")
        st.write(f"Neutral: {neu} ({int(neu/total*100)}%)")

        # WORD CLOUD
        st.subheader("☁️ Word Cloud")

        from wordcloud import WordCloud

        text = " ".join(df["Tweet"])
        wc = WordCloud(width=800, height=400).generate(text)

        st.image(wc.to_array())

        # INSIGHTS
        st.subheader("🧠 Insights")

        if pos > neg:
            st.success("Overall sentiment is Positive 😊")
        elif neg > pos:
            st.error("Overall sentiment is Negative 😡")
        else:
            st.info("Sentiment is Neutral 😐")