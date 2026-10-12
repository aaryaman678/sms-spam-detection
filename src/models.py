"""Classical machine learning models module for SMS Spam Detection.

This module provides model definitions, training routines, inference helpers,
and persistence utilities for scikit-learn classifiers.
"""

import os
from typing import Any, Dict, List

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

MODELS: Dict[str, Any] = {
    'naive_bayes': MultinomialNB(),
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'svm': LinearSVC(random_state=42, max_iter=2000),
}


def train_model(model_name: str, X_train: Any, y_train: Any) -> Any:
    """Get a model by name, fit it on the training data, and return the fitted model.

    Args:
        model_name: Name of the model ('naive_bayes', 'logistic_regression', or 'svm').
        X_train: Training features (array-like or sparse matrix).
        y_train: Training labels (array-like).

    Returns:
        Any: Fitted scikit-learn estimator.

    Raises:
        ValueError: If model_name is not found in the MODELS dictionary.
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {get_available_models()}"
        )

    model = MODELS[model_name]
    model.fit(X_train, y_train)
    return model


def predict(model: Any, X: Any) -> Any:
    """Generate predictions for input features using the given model.

    Args:
        model: Trained scikit-learn model supporting a predict method.
        X: Input features to predict on.

    Returns:
        Any: Predicted class labels from model.predict(X).
    """
    return model.predict(X)


def save_model(model: Any, filepath: str) -> None:
    """Save a trained model to disk using joblib.

    Automatically creates any missing parent directories.

    Args:
        model: Trained model to persist.
        filepath: Destination file path.
    """
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """Load a trained model from disk using joblib.

    Args:
        filepath: Path to the serialized model file.

    Returns:
        Any: Deserialized scikit-learn model.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)


def get_available_models() -> List[str]:
    """Return a list of available model names.

    Returns:
        List[str]: List of model name strings registered in the MODELS dictionary.
    """
    return list(MODELS.keys())
