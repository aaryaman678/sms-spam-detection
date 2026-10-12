"""Main training pipeline script for SMS Spam Detection NLP project.

Orchestrates data loading, text preprocessing, TF-IDF feature extraction,
training and evaluation of classical ML models, optional deep learning LSTM
training, and cross-model performance comparison with saved artifacts.

Usage:
    python train.py
"""

import os
import sys
import time

# Add project root to path so src imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src import data_loader
from src import evaluator
from src import models
from src import preprocessor
from src.feature_extractor import TfidfFeatureExtractor


def main() -> None:
    """Execute the end-to-end model training, evaluation, and serialization pipeline."""

    print("=" * 60)
    print("SMS Spam Detection - Training Pipeline")
    print("=" * 60)

    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # ── Step 1: Load Data ──────────────────────────────────────────
    print("\n--- Step 1: Load Data ---")
    df = data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.split_data(df)
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples : {len(X_test)}")

    # ── Step 2: Preprocess ─────────────────────────────────────────
    print("\n--- Step 2: Preprocess ---")
    X_train_clean = preprocessor.preprocess_series(X_train)
    X_test_clean = preprocessor.preprocess_series(X_test)
    print("  Preprocessing complete.")

    # ── Step 3: Extract Features (TF-IDF) ──────────────────────────
    print("\n--- Step 3: Extract Features ---")
    extractor = TfidfFeatureExtractor()
    X_train_vec = extractor.fit_transform(X_train_clean)
    X_test_vec = extractor.transform(X_test_clean)
    extractor.save(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    print(f"  TF-IDF features: {X_train_vec.shape[1]}")

    # ── Step 4: Train Classical Models ─────────────────────────────
    print("\n--- Step 4: Train Classical Models ---")
    all_results = []

    for model_name in models.get_available_models():
        print(f"\n  Training: {model_name}")
        start = time.time()

        # Train
        trained_model = models.train_model(model_name, X_train_vec, y_train)

        # Predict
        y_pred = models.predict(trained_model, X_test_vec)

        # Evaluate
        result = evaluator.evaluate_model(y_test, y_pred, model_name=model_name)
        all_results.append(result)

        # Confusion matrix
        cm_path = os.path.join(models_dir, f"{model_name}_confusion_matrix.png")
        evaluator.plot_confusion_matrix(y_test, y_pred, model_name=model_name, save_path=cm_path)

        # Save model
        model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
        models.save_model(trained_model, model_path)

        elapsed = time.time() - start
        print(f"  Saved {model_name} -> {model_path} ({elapsed:.2f}s)")

    # ── Step 5: Train LSTM ─────────────────────────────────────────
    print("\n--- Step 5: Train LSTM ---")
    try:
        from src import lstm_model

        # Train LSTM with preprocessed text
        trained_lstm, tokenizer, history = lstm_model.train_lstm(
            X_train_clean, y_train, X_test_clean, y_test
        )

        # Predict on test set
        X_test_seq, _ = lstm_model.prepare_sequences(X_test_clean, tokenizer)
        y_prob_lstm = trained_lstm.predict(X_test_seq)
        y_pred_lstm = (np.squeeze(y_prob_lstm) > 0.5).astype(int)

        # Evaluate
        lstm_result = evaluator.evaluate_model(y_test, y_pred_lstm, model_name="LSTM")
        all_results.append(lstm_result)

        # Confusion matrix
        lstm_cm_path = os.path.join(models_dir, "lstm_confusion_matrix.png")
        evaluator.plot_confusion_matrix(y_test, y_pred_lstm, model_name="LSTM", save_path=lstm_cm_path)

        # Save model and tokenizer
        lstm_model.save_lstm(
            trained_lstm,
            tokenizer,
            model_path=os.path.join(models_dir, "lstm_model.keras"),
            tokenizer_path=os.path.join(models_dir, "lstm_tokenizer.pkl"),
        )
        print("  LSTM model and tokenizer saved successfully.")

    except ImportError as exc:
        print(f"  [Warning] TensorFlow not available ({exc}). Skipping LSTM training.")
    except Exception as exc:
        print(f"  [Warning] Failed to train LSTM ({exc}). Skipping gracefully.")

    # ── Step 6: Compare Models ─────────────────────────────────────
    print("\n--- Step 6: Compare Models ---")
    if all_results:
        comparison_df = evaluator.compare_models(all_results)
        comp_path = os.path.join(models_dir, "model_comparison.png")
        evaluator.plot_model_comparison(comparison_df, save_path=comp_path)

    # ── Done ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training complete! All models saved to 'models/'.")
    print("Run the interactive web app with:")
    print("    streamlit run app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
