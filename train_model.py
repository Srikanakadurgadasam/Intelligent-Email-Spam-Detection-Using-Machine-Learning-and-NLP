import pandas as pd
import numpy as np
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels into numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):

    # Convert to lowercase
    text = text.lower()

    # Split words
    words = text.split()

    clean_words = []

    # Remove punctuation
    for word in words:
        if word.isalnum():
            clean_words.append(word)

    final_words = []

    # Remove stopwords and apply stemming
    for word in clean_words:
        if word not in stopwords.words('english'):
            final_words.append(ps.stem(word))

    return " ".join(final_words)

# Apply preprocessing
df['transformed_message'] = df['message'].apply(transform_text)

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_message']).toarray()

y = df['label'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save vectorizer
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model and vectorizer saved successfully!")
