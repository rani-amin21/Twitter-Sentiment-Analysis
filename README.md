# Twitter Sentiment Analysis  

An AI-powered web application that analyzes Twitter text and predicts sentiment as **Positive**, **Negative**, or **Neutral** using Machine Learning techniques.

🔗 **Live App:** https://twitter-sentiment-analysisapp.streamlit.app/

---

## 🚀 Features  

### 📝 Text Analysis Module  
- Enter tweet text manually  
- Text preprocessing (cleaning, stopword removal)  
- TF-IDF vectorization  
- Sentiment prediction (**Positive / Negative / Neutral**)  

### ⚡ Real-Time Prediction  
- Instant sentiment result  
- Fast and lightweight ML model  
- Simple UI using Streamlit  

---

## 🧠 Technologies Used  

- Python  
- Scikit-learn  
- Pandas & NumPy  
- TF-IDF Vectorization  
- Machine Learning Models  
  - Logistic Regression  
  - Naive Bayes  
- Pickle (Model Saving)  
- Streamlit  

---

## 🏗 How It Works  

1. Input tweet text  
2. Text preprocessing (lowercase, remove punctuation, stopwords)  
3. Convert text into vectors using TF-IDF  
4. Load trained ML model  
5. Predict sentiment  
6. Display result to user  

---

## 📊 Model Details  

- **Vectorizer:** TF-IDF  
- **Models Used:**  
  - Logistic Regression  
  - Naive Bayes  
- Model saved as `.pkl` file  

---

## 📊 Sentiment Output  

- Positive 😊  
- Negative 😡  
- Neutral 😐  

---

## 🔐 Security  

- No user data is stored  
- Model runs locally or on server  
- Safe text processing  

---

## 💡 Future Improvements  

- Add live Twitter API integration  
- Improve UI design  
- Add sentiment visualization charts  
- Add user history tracking  
