import streamlit as st
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk, re, string

# Download only first time
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load model & vectorizer
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Preprocessing function (same as training)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", " ", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Streamlit UI
st.title("🛒 Amazon Review Sentiment Analyzer")
st.write("Enter a product review and find out if it is **Positive or Negative**")

# Text input
user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        if pred == 1:
            st.success("✅ Sentiment: Positive Review")
        else:
            st.error("❌ Sentiment: Negative Review")
    else:
        st.warning("Please enter a review first.")
