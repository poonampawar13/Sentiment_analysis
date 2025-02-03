import numpy as np
import pandas as pd
import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load("notebook/model.pkl")
vectorizer = joblib.load("notebook/TfidfVectorizer.pkl")


# Function to clean text
def remove_punch(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Streamlit UI
st.set_page_config(page_title='SENTIMENT ANALYSIS', layout='wide')
st.title('SENTIMENT ANALYSIS')
st.subheader("Enter a sentence, and the model will predict its sentiment.")

# Input text box
user_input = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_input:
        # Clean and transform input text
        cleaned_text = remove_punch(user_input.lower())
        transformed_text = vectorizer.transform([cleaned_text])

        # Predict sentiment
        prediction = model.predict(transformed_text)[0]

        # Display result
        sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
        st.write(f"**Sentiment:** {sentiment}")
    else:
        st.warning("Please enter some text to analyze.")