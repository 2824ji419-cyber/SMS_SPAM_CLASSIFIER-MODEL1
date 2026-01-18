import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


import os

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()

# Load models
# Using absolute path logic to be safe if run from different cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(vectorizer_path, "rb") as file:
    tfidf = pickle.load(file)

with open(model_path, "rb") as file:
    model = pickle.load(file)

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

st.title("📩 SMS Spam Classifier")

msg = st.text_area("Enter SMS Message")

if st.button("Predict"):
    processed_msg = preprocess(msg)
    # Vectorize the input
    vector_input = tfidf.transform([processed_msg])
    # Predict
    prediction = model.predict(vector_input)[0]

    if prediction == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")

