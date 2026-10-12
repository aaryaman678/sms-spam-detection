"""Inference and explainability module for SMS Spam Detection.

Provides functions to classify raw SMS text messages using either trained
classical ML models (Naive Bayes, Logistic Regression, SVM) or a deep
learning LSTM model, as well as keyword attribution for model explainability.

Usage:
    from src.predict import predict_sms, predict_all_models, explain_keywords
    result = predict_sms("You won a free prize!", model_type="naive_bayes")
"""

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is present in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def predict_sms(message: str, model_type: str = "naive_bayes") -> dict:
    """Classify an SMS message as 'spam' or 'ham' using a specified model.

    Preprocesses the input text, loads the requested model artifact and
    feature representations, generates predictions, and returns a result
    dictionary with the label and confidence score.

    Args:
        message: Raw SMS text string to classify.
        model_type: Model identifier - one of 'naive_bayes',
            'logistic_regression', 'svm', or 'lstm'. Defaults to 'naive_bayes'.

    Returns:
        dict with keys:
            - 'message': Original input message
            - 'cleaned_text': Preprocessed message text
            - 'prediction': 'spam' or 'ham'
            - 'confidence': float between 0 and 1, or None if unavailable
            - 'model': The model_type string used
    """
    from src.preprocessor import preprocess_text

    cleaned_text = preprocess_text(message)
    models_dir = os.path.join(PROJECT_ROOT, "models")

    if model_type.lower() == "lstm":
        return _predict_lstm(message, cleaned_text, models_dir)
    else:
        return _predict_classical(message, cleaned_text, models_dir, model_type)


def predict_all_models(message: str) -> Dict[str, dict]:
    """Run inference across all available classical models for direct side-by-side comparison.

    Args:
        message: Raw SMS text string to classify.

    Returns:
        Dictionary mapping model keys ('naive_bayes', 'logistic_regression', 'svm')
        to their respective prediction results.
    """
    results = {}
    for m in ["naive_bayes", "logistic_regression", "svm"]:
        try:
            results[m] = predict_sms(message, model_type=m)
        except Exception as e:
            results[m] = {"error": str(e), "prediction": "error", "confidence": None, "model": m}
    return results


def explain_keywords(message: str, top_n: int = 8) -> List[Dict[str, Any]]:
    """Identify which words in the message contributed most to spam or ham prediction.

    Uses the feature weights from the trained Logistic Regression classifier and
    TF-IDF vocabulary to assign importance scores to matched tokens.

    Args:
        message: Raw SMS message.
        top_n: Maximum number of influential tokens to return.

    Returns:
        List of dicts: [{'word': str, 'weight': float, 'impact': 'spam'|'ham'}]
        sorted by absolute weight descending.
    """
    import joblib
    from src.preprocessor import preprocess_text

    models_dir = os.path.join(PROJECT_ROOT, "models")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    lr_path = os.path.join(models_dir, "logistic_regression_model.pkl")

    if not os.path.exists(vec_path) or not os.path.exists(lr_path):
        return []

    try:
        vec = joblib.load(vec_path)
        lr = joblib.load(lr_path)
    except Exception:
        return []

    feature_names = np.array(vec.get_feature_names_out())
    coefs = lr.coef_[0]

    cleaned_text = preprocess_text(message)
    tokens = set(cleaned_text.split())

    contributions = []
    for token in tokens:
        if token in vec.vocabulary_:
            idx = vec.vocabulary_[token]
            weight = float(coefs[idx])
            contributions.append({
                "word": token,
                "weight": round(weight, 4),
                "impact": "Spam Trigger" if weight > 0 else "Legitimate Indicator",
                "abs_weight": abs(weight),
            })

    # Sort by absolute weight descending
    contributions.sort(key=lambda x: x["abs_weight"], reverse=True)
    return contributions[:top_n]


def _predict_lstm(message: str, cleaned_text: str, models_dir: str) -> dict:
    """Run inference with the LSTM model."""
    from src.lstm_model import load_lstm, prepare_sequences

    model_path = os.path.join(models_dir, "lstm_model.keras")
    tokenizer_path = os.path.join(models_dir, "lstm_tokenizer.pkl")

    model, tokenizer = load_lstm(model_path, tokenizer_path)

    padded_seq, _ = prepare_sequences([cleaned_text], tokenizer=tokenizer)
    prob = float(np.squeeze(model.predict(padded_seq)))

    prediction = 1 if prob > 0.5 else 0
    confidence = prob if prediction == 1 else 1.0 - prob

    return {
        "message": message,
        "cleaned_text": cleaned_text,
        "prediction": "spam" if prediction == 1 else "ham",
        "confidence": round(float(confidence), 4),
        "model": "lstm",
    }


def _predict_classical(
    message: str, cleaned_text: str, models_dir: str, model_type: str
) -> dict:
    """Run inference with a classical ML model (NB, LR, SVM)."""
    from src.feature_extractor import TfidfFeatureExtractor
    from src.models import load_model

    # Load TF-IDF vectorizer
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    extractor = TfidfFeatureExtractor()
    extractor.load(vectorizer_path)

    # Load trained model
    model_path = os.path.join(models_dir, f"{model_type}_model.pkl")
    model = load_model(model_path)

    # Transform and predict
    features = extractor.transform([cleaned_text])
    raw_pred = int(model.predict(features)[0])
    prediction = 1 if raw_pred == 1 else 0

    # Calculate confidence score
    confidence: Optional[float] = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(features)[0]
            prob_predicted = float(probas[1] if prediction == 1 else probas[0])
            confidence = prob_predicted
        except Exception:
            confidence = None
    elif hasattr(model, "decision_function"):
        try:
            decision_val = float(model.decision_function(features)[0])
            prob_spam = 1.0 / (1.0 + np.exp(-decision_val))
            confidence = float(prob_spam if prediction == 1 else 1.0 - prob_spam)
        except Exception:
            confidence = None

    return {
        "message": message,
        "cleaned_text": cleaned_text,
        "prediction": "spam" if prediction == 1 else "ham",
        "confidence": round(confidence, 4) if confidence is not None else None,
        "model": model_type,
    }


if __name__ == "__main__":
    sample_ham = "Hey, are we still meeting for coffee this afternoon at 3?"
    sample_spam = (
        "URGENT! You have won a 1,000 cash prize or a holiday! "
        "Call 09050000327 to claim NOW. T&Cs apply."
    )

    print("=" * 60)
    print("SMS Spam Detection - Prediction Demo")
    print("=" * 60)

    for test_model in ["naive_bayes", "logistic_regression", "svm"]:
        print(f"\n--- Model: {test_model} ---")
        try:
            for label, msg in [("HAM", sample_ham), ("SPAM", sample_spam)]:
                result = predict_sms(msg, model_type=test_model)
                conf = f"{result['confidence']:.4f}" if result["confidence"] else "N/A"
                print(f"  [{label}] -> {result['prediction'].upper()} (conf: {conf})")
        except FileNotFoundError:
            print("  Models not found. Run 'python train.py' first.")
        except Exception as err:
            print(f"  Error: {err}")

    print("\n--- Feature Attribution / Explainability Demo ---")
    explained = explain_keywords(sample_spam)
    for item in explained:
        print(f"  Word: '{item['word']:<10}' | Weight: {item['weight']:+.4f} ({item['impact']})")
