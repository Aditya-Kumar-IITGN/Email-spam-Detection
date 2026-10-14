import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Page configuration
st.set_page_config(
    page_title="Email & SMS Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# Auto-download required NLTK datasets for cloud environments (Streamlit Cloud, Render, etc.)
for pkg in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

lemma = WordNetLemmatizer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(lemma.lemmatize(i))
    return " ".join(y)

# Cache model and vectorizer loading
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tfidf = pickle.load(open(os.path.join(base_dir, 'vectorizer.pkl'), 'rb'))
    model = pickle.load(open(os.path.join(base_dir, 'model.pkl'), 'rb'))
    return tfidf, model

tfidf, model = load_artifacts()

# App Header
st.title("🛡️ Email & SMS Spam Classifier")
st.markdown("Enter any email or SMS text message below to classify it as **Legitimate (Ham)** or **Spam**.")

# Quick test samples in an expander
with st.expander("💡 Click to view sample messages to test"):
    st.markdown("""
    * **Spam Sample**: `WINNER!! You have been selected to receive a £900 prize reward! Call 09061701461 to claim. Claim code KL341.`
    * **Ham Sample**: `Hey, are we still meeting at the library after class today? Let me know.`
    """)

input_sms = st.text_area("Enter your message:", height=140, placeholder="Paste or type your message here...")

if st.button("Classify Message", type="primary"):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message before classifying.")
    else:
        with st.spinner("Analyzing text..."):
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. Predict
            result = model.predict(vector_input)[0]

        # 4. Display Result
        st.write("---")
        if result == 1:
            st.error("🚨 **Prediction: SPAM**\n\nThis message exhibits common spam indicators (promotional language, artificial urgency, or financial incentives).")
        else:
            st.success("✅ **Prediction: NOT SPAM (Legitimate)**\n\nThis message appears to be safe personal or regular communication.")