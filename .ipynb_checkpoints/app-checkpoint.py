import joblib
import streamlit as st
import numpy as np
import re

#  Load model, tfidf and label_encoder 
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("logistic_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return tfidf, model, label_encoder

tfidf, model, label_encoder = load_artifacts()

#  Text Cleaning 
def clean_text(text):
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\+?\d[\d -]{8,}\d', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

#  Prediction 
def predict_role(resume_text):
    cleaned = clean_text(resume_text)
    X = tfidf.transform([cleaned])
    pred = model.predict(X)[0]
    if isinstance(pred, (int, np.integer)):
        return label_encoder.inverse_transform([pred])[0]
    return pred

#  Streamlit UI 
st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("Resume Screener")

uploaded_file = st.file_uploader("Upload resume (PDF / TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.size > 200 * 1024 * 1024:  
        st.error("File size exceeds 200 MB limit.")
    else:
        # Extract text
        text = ""
        if uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                text = "\n".join([p.extract_text() or "" for p in reader.pages])
            except Exception:
                st.error("Error reading PDF. Please try another file.")
        else:
            text = uploaded_file.read().decode("utf-8", errors="ignore")

        # Predict
        if text.strip() and st.button("Predict Job Role"):
            role = predict_role(text)
            st.success(f"**Predicted Job Role:** {role}")
        elif not text.strip():
            st.warning(" No text could be extracted from the file.")
