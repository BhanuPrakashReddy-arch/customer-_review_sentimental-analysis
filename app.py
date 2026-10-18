
import joblib
import streamlit as st
from pathlib import Path

# Load the trained model and TF-IDF vectorizer
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "models" / "sentiment_model.pkl"
)
vectorizer = joblib.load(
    BASE_DIR / "models" / "tfidf_vectorizer.pkl"
)

# Web application
st.title("Customer Review Sentiment Analysis")
st.write("Enter a customer review to predict its sentiment.")

review = st.text_area("Enter your customer review:")

if st.button("Analyze Sentiment"):
    if review.strip():
        features = vectorizer.transform([review])
        prediction = model.predict(features)[0]

        st.success(f"Predicted Sentiment: {prediction.title()}")
    else:
        st.warning("Please enter a customer review.")
