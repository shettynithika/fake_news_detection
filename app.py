import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")

user_input = st.text_area("Enter the news text to check:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vector_input = vectorizer.transform([user_input])
        prediction = model.predict(vector_input)
        label = "REAL" if prediction[0] == 1 else "FAKE"
        st.success(f"Prediction: {label}")
