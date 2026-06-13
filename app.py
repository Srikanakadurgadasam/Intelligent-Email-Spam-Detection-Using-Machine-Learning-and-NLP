import streamlit as st
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing
def transform_text(text):

    text = text.lower()

    words = text.split()

    clean_words = []

    for word in words:
        if word.isalnum():
            clean_words.append(word)

    final_words = []

    for word in clean_words:
        if word not in stopwords.words('english'):
            final_words.append(ps.stem(word))

    return " ".join(final_words)

# Streamlit UI
st.title("Spam Email Classifier")

input_sms = st.text_area("Enter your message")

if st.button('Predict'):

    # Preprocess input
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = vectorizer.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Output
    if result == 1:
        st.header("Spam Message")
    else:
        st.header("Not Spam Message")
