import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import re

# Page config
st.set_page_config(page_title="Gender Text Classifier", page_icon="👨‍🦰👧‍", layout="centered")

# App Title
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'> Gender Prediction from Text</h1>
    <p style='text-align: center;'>A simple NLP model that predicts whether text is written by a <b>Male</b> or <b>Female</b>.</p>
    <hr>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("genderdata.csv")
    df = df.dropna()
    df['gender'] = df['gender'].str.strip().str.title()  # Normalize to 'Male'/'Female'
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))  # Basic cleaning
    return df

df = load_data()

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['gender']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)



# Input box
st.markdown("### ✏️ Enter Text to Predict Gender:")
user_input = st.text_area("Enter a sentence:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Clean input
        clean_input = re.sub(r'[^\w\s]', '', user_input.lower())
        transformed = vectorizer.transform([clean_input])
        prediction = model.predict(transformed)[0]
        st.success(f"🎯 Predicted Gender: **{prediction}**")


