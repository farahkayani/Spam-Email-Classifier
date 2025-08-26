# 📧 Spam Email Classifier

A machine learning pipeline for classifying emails as **spam** or **not spam** using Logistic Regression, Naive Bayes, and SVM.  
Includes **CLI tools** for training & prediction, **auto model selection & threshold tuning**, and a **Streamlit web app** for interactive testing.

---

## 🚀 Features
- Train models (Naive Bayes, Logistic Regression, SVM)
- Automatic threshold tuning for best F1-score
- Model selection based on validation performance
- Evaluate on held-out test set with detailed metrics & plots
- Save & load trained models and metadata
- CLI for training and prediction
- Streamlit app for interactive spam detection

---

## 📂 Repository Structure
```
├── data/ # Dataset (sample email data)
├── artifacts/ # Saved models, metrics, plots
├── spam_email_pipeline.py # CLI script (train/predict)
├── train_and_select.py # Model training & selection logic
├── streamlit_app.py # Web app for spam detection
├── requirements.txt # Dependencies
└── README.md # Project documentation
```

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-email-classifier.git
   cd spam-email-classifier
   ```
2. Create and activate virtual environment:
   ```
   python -m venv lora-env
   source lora-env/bin/activate   # Mac/Linux
   lora-env\Scripts\activate      # Windows

   ```
   
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
---

## 🏋️‍♀️ **Training**

To train and save the best model:
```
python spam_email_pipeline.py --train data/emails.csv --output_dir artifacts
```

This will:

Train models (NB, LR, SVM)

Select the best one using validation F1

Save the final model and metadata in artifacts/

---

## 🌐 **Streamlit App**

To launch the interactive app:
```
streamlit run streamlit_app.py
```

Paste any email text and get Spam Probability (%) instantly.

---

## 📊 **Evaluation**

During training, the pipeline saves:

Confusion matrix

Precision, Recall, F1

ROC & PR curves

Best threshold

Report (report.txt in artifacts/)

---

## 📦 **Requirements**

Main libraries:
```
scikit-learn

pandas

numpy

joblib

matplotlib

streamlit
```

Install via:
```

pip install -r requirements.txt
```

---

## 📝 **Example Spam Email**
"Congratulations! You have been selected to receive a FREE gift card worth $500. 
Click here to claim your prize now!"


This should be detected as Spam.

---

## 📌 **Project Info**
```
Author: Farah Kayani

License: MIT

Description: A complete spam email detection system with ML models, CLI, and Streamlit app.
```
