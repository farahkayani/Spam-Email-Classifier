# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 23:46:24 2025

@author: farah
"""
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, roc_curve, auc, accuracy_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


# 1) Text preprocessing

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = _URL_RE.sub(" ", s)
    s = _HTML_RE.sub(" ", s)
    s = re.sub(r"\d+", " ", s)               # drop numbers
    s = _NON_ALPHA_RE.sub(" ", s)            # keep only letters & spaces
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


# 2) Utilities for labels & loading

def normalize_labels(series: pd.Series) -> pd.Series:
    """
    Maps labels to {0,1} where 1=spam, 0=ham.
    Accepts:
      - numeric {0,1}
      - strings 'spam'/'ham' (case-insensitive)
      - common variants: 'spam'/'not spam' or 'ham'
    """
    s = series.copy()
    # If already numeric and likely binary:
    if pd.api.types.is_numeric_dtype(s):
        # ensure values are 0/1
        uniques = set(pd.unique(s))
        if uniques <= {0, 1}:
            return s.astype(int)
        # try to coerce other numeric encodings
        return s.apply(lambda x: 1 if int(x) == 1 else 0).astype(int)

    s = s.astype(str).str.strip().str.lower()
    def map_label(x: str) -> int:
        if x in {"spam", "1", "true", "yes"}:
            return 1
        if x in {"ham", "0", "false", "no", "not spam", "legit", "legitimate"}:
            return 0
        # fallback heuristic
        return 1 if "spam" in x else 0
    return s.apply(map_label).astype(int)


def load_dataset(path: str, text_col: Optional[str], label_col: Optional[str]) -> Tuple[pd.Series, pd.Series]:
    # Load only label + text, skip broken rows
    df = pd.read_csv(
        path,
        usecols=[0, 1],
        names=["label", "text"],
        header=0,
        encoding="latin-1",
        on_bad_lines="skip"
    )

    # Auto-detect columns if not provided
    if text_col is None:
        candidates = [c for c in df.columns if c.lower() in {"text", "message", "email", "email_text", "content", "sms"}]
        if not candidates:
            raise ValueError(f"Could not auto-detect text column in {list(df.columns)}. Use --text-col.")
        text_col = candidates[0]
    if label_col is None:
        candidates = [c for c in df.columns if c.lower() in {"label", "target", "class", "category", "spam"}]
        if not candidates:
            raise ValueError(f"Could not auto-detect label column in {list(df.columns)}. Use --label-col.")
        label_col = candidates[0]

    X = df[text_col].astype(str).map(clean_text)
    y = normalize_labels(df[label_col])
    return X, y

def prepare_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Apply SMOTE only on training set
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train.values.reshape(-1,1), y_train)
    
    # Flatten back
    X_train_res = X_train_res.ravel()
    return X_train_res, X_test, y_train_res, y_test

# 3) Model builders


def build_vectorizer(max_features: int, ngram_min: int, ngram_max: int, min_df: int, max_df: float) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=None,          # we already cleaned text explicitly
        tokenizer=None,
        lowercase=False,            # already lowered
        stop_words="english",
        ngram_range=(ngram_min, ngram_max),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )

def build_nb_pipeline(vec: TfidfVectorizer) -> Pipeline:
    return Pipeline([("tfidf", vec), ("clf", MultinomialNB())])

def build_lr_pipeline(vec: TfidfVectorizer) -> Pipeline:
    # 'liblinear' is solid for high-dimensional sparse text
    return Pipeline([("tfidf", vec), ("clf", LogisticRegression(max_iter=2000, solver="liblinear", class_weight=None))])

def build_svm_pipeline(vec: TfidfVectorizer) -> Pipeline:
    return Pipeline([("tfidf", vec), ("clf", LinearSVC())])


# 4) Threshold tuning (for NB/LR)

@dataclass
class ThresholdResult:
    threshold: float
    f1: float
    precision: float
    recall: float

def tune_threshold(pipe: Pipeline, X_val, y_val) -> ThresholdResult:
    """
    Finds the best probability threshold (0..1) maximizing F1 on validation data.
    Works only if the pipeline supports predict_proba.
    """
    if not hasattr(pipe, "predict_proba"):
        # e.g., LinearSVC – return a neutral threshold indicator
        preds = pipe.predict(X_val)
        return ThresholdResult(threshold=-1.0,
                               f1=f1_score(y_val, preds),
                               precision=precision_score(y_val, preds, zero_division=0),
                               recall=recall_score(y_val, preds, zero_division=0))
    probs = pipe.predict_proba(X_val)[:, 1]
    best = ThresholdResult(threshold=0.5, f1=-1.0, precision=0.0, recall=0.0)
    for t in np.linspace(0.05, 0.95, 19):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        if f1 > best.f1:
            best = ThresholdResult(threshold=float(t), f1=float(f1), precision=float(p), recall=float(r))
    return best


def predict_with_threshold(pipe: Pipeline, X, threshold: float) -> np.ndarray:
    if threshold < 0 or not hasattr(pipe, "predict_proba"):
        return pipe.predict(X)
    probs = pipe.predict_proba(X)[:, 1]
    return (probs >= threshold).astype(int)


# 5) Evaluation & plotting


def evaluate(pipe: Pipeline, X_test, y_test, threshold: float, out_dir: str, model_name: str) -> Dict:
    preds = predict_with_threshold(pipe, X_test, threshold)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    # Save confusion matrix plot
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Ham (0)", "Spam (1)"])
    plt.yticks(tick_marks, ["Ham (0)", "Spam (1)"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    cm_path = os.path.join(out_dir, f"{model_name}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # PR and ROC (only if probabilities available)
    pr_path, roc_path, auc_pr, auc_roc = None, None, None, None
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(X_test)[:, 1]
        # PR curve
        precision, recall, _ = precision_recall_curve(y_test, probs)
        plt.figure()
        plt.plot(recall, precision)
        plt.title(f"Precision-Recall Curve – {model_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        pr_path = os.path.join(out_dir, f"{model_name}_pr_curve.png")
        plt.tight_layout()
        plt.savefig(pr_path, dpi=150)
        plt.close()
        auc_pr = auc(recall, precision)
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title(f"ROC Curve – {model_name}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        roc_path = os.path.join(out_dir, f"{model_name}_roc_curve.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=150)
        plt.close()
        auc_roc = auc(fpr, tpr)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    return {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc_pr": float(auc_pr) if auc_pr is not None else None,
        "auc_roc": float(auc_roc) if auc_roc is not None else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "plots": {
            "confusion_matrix": cm_path,
            "pr_curve": pr_path,
            "roc_curve": roc_path
        }
    }


# 6) Training orchestration


def train_and_select(
    X: pd.Series,
    y: pd.Series,
    out_dir: str,
    max_features: int,
    ngram_min: int,
    ngram_max: int,
    min_df: int,
    max_df: float,
    test_size: float,
    random_state: int,
    train_models: List[str],
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    # Split -> Test; then split train into train/val for threshold tuning & model selection
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # Vectorizer (shared instance for validation stage)
    vec = build_vectorizer(max_features, ngram_min, ngram_max, min_df, max_df)

    # Candidate models dictionary
    models = {
        "nb": build_nb_pipeline(vec),
        "lr": build_lr_pipeline(vec),
        "svm": build_svm_pipeline(vec)
    }

    # Train + tune each requested model
    candidates = {}
    for name in train_models:
        if name not in models:
            continue
        pipe = models[name]
        pipe.fit(X_tr, y_tr)
        thr = tune_threshold(pipe, X_val, y_val) if name in {"nb", "lr"} else tune_threshold(pipe, X_val, y_val)
        candidates[name] = (pipe, thr)

    # Pick best by validation F1
    best_name, best_pipe, best_thr, best_f1 = None, None, None, -1.0
    selection_summary = {}
    for name, (pipe, thr) in candidates.items():
        selection_summary[name] = {
            "val_threshold": thr.threshold,
            "val_precision": thr.precision,
            "val_recall": thr.recall,
            "val_f1": thr.f1
        }
        if thr.f1 > best_f1:
            best_name, best_pipe, best_thr, best_f1 = name, pipe, thr.threshold, thr.f1

    # Refit best model on full training set
    vec_full = build_vectorizer(max_features, ngram_min, ngram_max, min_df, max_df)
    final_pipe = models[best_name] = (
        build_nb_pipeline(vec_full) if best_name == "nb"
        else build_lr_pipeline(vec_full) if best_name == "lr"
        else build_svm_pipeline(vec_full)
    )
    final_pipe.fit(X_train, y_train)

    # Re-tune threshold if model supports probabilities
    final_thr = best_thr
    if hasattr(final_pipe, "predict_proba") and best_name in {"nb", "lr"}:
        tr = tune_threshold(final_pipe, X_val, y_val)
        final_thr = tr.threshold

    # Evaluate on held-out test set
    results = evaluate(final_pipe, X_test, y_test, final_thr, out_dir, f"best_{best_name}")

    # Save model
    model_path = os.path.join(out_dir, "best_model.joblib")
    joblib.dump(final_pipe, model_path)

    # Save metadata
    meta = {
        "best_model": best_name,
        "chosen_threshold": final_thr,
        "selection_summary": selection_summary,
        "vectorizer": {
            "max_features": max_features,
            "ngram_range": [ngram_min, ngram_max],
            "min_df": min_df,
            "max_df": max_df,
            "stop_words": "english",
        },
        "splits": {"test_size": test_size, "random_state": random_state},
        "metrics_test": results
    }
    meta_path = os.path.join(out_dir, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Quick report
    text_report = (
        f"Best model: {best_name}\n"
        f"Chosen threshold: {final_thr}\n"
        f"Test Accuracy: {results['accuracy']:.4f}\n"
        f"Test Precision: {results['precision']:.4f}\n"
        f"Test Recall: {results['recall']:.4f}\n"
        f"Test F1: {results['f1']:.4f}\n"
        f"AUC-PR: {results['auc_pr']}\n"
        f"AUC-ROC: {results['auc_roc']}\n"
        f"Confusion Matrix: {results['confusion_matrix']}\n"
    )
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(text_report)

    print("\n=== Training complete ===")
    print(text_report)
    print(f"Artifacts saved to: {out_dir}")
    print(f"- Model: {model_path}")
    print(f"- Meta:  {meta_path}")
    print(f"- Plots: {results['plots']}")

    return meta



# 7) Prediction


def predict_single(text: str, out_dir: str) -> Dict:
    model_path = os.path.join(out_dir, "best_model.joblib")
    meta_path = os.path.join(out_dir, "model_meta.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Trained model/meta not found. Please run training first.")

    pipe = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    thr = meta.get("chosen_threshold", -1.0)

    cleaned = clean_text(text)
    pred = predict_with_threshold(pipe, [cleaned], thr)[0]
    # probability if available
    prob = None
    if hasattr(pipe, "predict_proba"):
        prob = float(pipe.predict_proba([cleaned])[:, 1][0])

    result = {
        "input": text,
        "cleaned": cleaned,
        "prediction": int(pred),
        "label": "spam" if pred == 1 else "ham",
        "prob_spam": prob
    }
    print(json.dumps(result, indent=2))
    return result

def predict_email(text: str, model_path: str, meta_path: str):
    # Load model and metadata
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    threshold = meta.get("chosen_threshold", 0.5)

    # Predict probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba([text])[0, 1]
    else:
        # For SVM without probabilities, fallback to decision_function
        prob = model.decision_function([text])[0]
        prob = 1 / (1 + np.exp(-prob))  # convert to probability

    # Apply threshold
    label = "SPAM" if prob >= threshold else "HAM"
    print(f"Input: {text}")
    print(f"Predicted: {label} (score: {prob:.2f}, threshold: {threshold:.2f})")
    return label, prob


# 8) CLI

def classify_email(email_text, model_path="artifacts/best_model.joblib", meta_path="artifacts/model_meta.json"):
    # Load model
    model = joblib.load(model_path)

    # Load metadata (like vectorizer info, threshold, etc.)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    threshold = meta.get("threshold", 0.5)

    # Predict probability
    proba = model.predict_proba([email_text])[0][1]

    # Apply threshold
    label = 1 if proba >= threshold else 0

    return {
        "text": email_text,
        "predicted_label": "spam" if label == 1 else "ham",
        "probability_spam": proba
    }

def parse_args():
    ap = argparse.ArgumentParser(description="Spam Email Classification – End-to-End")
    ap.add_argument("--data", type=str, default=None, help="Path to CSV with text + label.")
    ap.add_argument("--text-col", type=str, default=None, help="Text column name (auto-detect if omitted).")
    ap.add_argument("--label-col", type=str, default=None, help="Label column name (auto-detect if omitted).")
    ap.add_argument("--output-dir", type=str, default="artifacts", help="Where to save models/plots/metrics.")

    # Vectorizer
    ap.add_argument("--max-features", type=int, default=30000)
    ap.add_argument("--ngram-min", type=int, default=1)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-df", type=float, default=0.9)

    # Splits
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)

    # Models to try
    ap.add_argument("--models", type=str, default="nb,lr,svm",
                    help="Comma-separated subset of {nb,lr,svm}")

    # Prediction mode
    ap.add_argument("--predict", type=str, default=None, help='Predict single email text (quote the string).')

    return ap.parse_args()


def main():
    args = parse_args()   # use the unified parser

    print("DEBUG args:", args)   # optional, for debugging

    if args.predict:  # ---------- Prediction mode ----------
        result = predict_single(args.predict, args.output_dir)
        print("\n=== Prediction Result ===")
        print(f"Email: {result['input']}")
        print(f"Cleaned: {result['cleaned']}")
        print(f"Predicted label: {result['label']}")
        if result['prob_spam'] is not None:
            print(f"Probability spam: {result['prob_spam']:.4f}")
        return

    if args.data:  # ---------- Training mode ----------
        X, y = load_dataset(args.data, args.text_col, args.label_col)
        train_and_select(
            X=X,
            y=y,
            out_dir=args.output_dir,
            max_features=args.max_features,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            min_df=args.min_df,
            max_df=args.max_df,
            test_size=args.test_size,
            random_state=args.random_state,
            train_models=[m.strip() for m in args.models.split(",") if m.strip() in {"nb", "lr", "svm"}],
        )
        return

    raise ValueError("Please provide either --data for training or --predict for prediction.")

if __name__ == "__main__":
    main()
