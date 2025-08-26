# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200

def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    score = model.predict(padded, verbose=0)[0][0]
    sentiment = "Positive 😊" if score >= 0.5 else "Negative 😞"
    return sentiment, score

# Streamlit UI
st.set_page_config(page_title="🎬 Sentiment Analyzer", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below and see if it's **Positive** or **Negative**.")

user_review = st.text_area("Your Review", height=150)

if st.button("Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        sentiment, score = predict_sentiment(user_review)
        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Prediction Score:** {score:.4f}")
