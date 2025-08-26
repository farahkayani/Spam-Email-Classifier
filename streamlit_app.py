# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 17:41:07 2025

@author: farah
"""

# streamlit_app.py
import streamlit as st
import os
import joblib
import json
from spam_email_pipeline import clean_text, predict_with_threshold
from spam_email_pipeline import train_and_select, load_dataset

# === Paths ===
MODEL_PATH = "artifacts/best_model.joblib"
META_PATH = "artifacts/model_meta.json"

st.set_page_config(page_title="Spam Email Classifier", layout="centered")

st.title("📧 Spam Email Classifier")
st.write("Train a spam classifier or test an email below.")

# --- Sidebar for training ---
st.sidebar.header("🔧 Train Model")
train_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
if st.sidebar.button("Train Model"):
    if train_file:
        # Save uploaded file temporarily
        tmp_path = "uploaded.csv"
        with open(tmp_path, "wb") as f:
            f.write(train_file.read())
        st.sidebar.write("✅ File uploaded. Training...")

        # Train model
        X, y = load_dataset(tmp_path, None, None)
        train_and_select(
            X, y,
            out_dir="artifacts",
            max_features=30000,
            ngram_min=1,
            ngram_max=2,
            min_df=2,
            max_df=0.9,
            test_size=0.2,
            random_state=42,
            train_models=["nb", "lr", "svm"]
        )
        st.sidebar.success("Training complete! Model saved in `artifacts/`.")
    else:
        st.sidebar.error("Please upload a dataset CSV first.")

# --- Email classification form ---
st.subheader("✉️ Test an Email")
email_text = st.text_area("Enter email text:", height=150)

if st.button("Classify Email"):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        st.error("No trained model found. Please train a model first from the sidebar.")
    else:
        # Load model + metadata
        pipe = joblib.load(MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        thr = meta.get("chosen_threshold", 0.5)

        cleaned = clean_text(email_text)
        pred = predict_with_threshold(pipe, [cleaned], thr)[0]

        label = "🚨 Spam" if pred == 1 else "✅ Ham (Not Spam)"
        st.markdown(f"**Prediction:** {label}")

        if hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba([cleaned])[0][1]
            st.metric("Probability of Spam", f"{prob:.2%}")
