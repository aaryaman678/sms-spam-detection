"""TF-IDF feature extraction module for SMS Spam Detection.

This module provides the TfidfFeatureExtractor class to convert text messages
into TF-IDF sparse matrix representations.
"""

import os
from typing import List, Optional, Tuple

import joblib
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfFeatureExtractor:
    """TF-IDF feature extractor for text classification.

    Attributes:
        max_features (Optional[int]): Maximum number of features (vocabulary size).
        ngram_range (Tuple[int, int]): Lower and upper boundary of the range of n-values for n-grams.
        vectorizer (Optional[TfidfVectorizer]): Scikit-learn TF-IDF vectorizer instance.
    """

    def __init__(
        self,
        max_features: Optional[int] = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
    ) -> None:
        """Initialize the TF-IDF feature extractor.

        Args:
            max_features: Maximum number of features to extract. Defaults to 5000.
            ngram_range: The lower and upper boundary of n-values for n-grams. Defaults to (1, 2).
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer: Optional[TfidfVectorizer] = None

    def fit_transform(self, X_train: pd.Series) -> spmatrix:
        """Create a TfidfVectorizer with configured parameters, fit on X_train, and return the transformed matrix.

        Args:
            X_train: Training text data (pandas Series of strings).

        Returns:
            scipy.sparse matrix: Transformed TF-IDF sparse matrix.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        return self.vectorizer.fit_transform(X_train)

    def transform(self, X: pd.Series) -> spmatrix:
        """Transform text data into TF-IDF features using the fitted vectorizer.

        Args:
            X: Text data to transform (pandas Series of strings).

        Returns:
            scipy.sparse matrix: Transformed TF-IDF sparse matrix.

        Raises:
            ValueError: If the vectorizer has not been fitted or loaded yet.
        """
        if self.vectorizer is None:
            raise ValueError(
                "Vectorizer has not been fitted or loaded. Call 'fit_transform' or 'load' first."
            )
        return self.vectorizer.transform(X)

    def save(self, filepath: str = 'models/tfidf_vectorizer.pkl') -> None:
        """Save the fitted vectorizer to disk using joblib.

        Creates any missing parent directories automatically.

        Args:
            filepath: Destination file path. Defaults to 'models/tfidf_vectorizer.pkl'.

        Raises:
            ValueError: If the vectorizer is not fitted yet.
        """
        if self.vectorizer is None:
            raise ValueError("No fitted vectorizer to save. Call 'fit_transform' or 'load' first.")

        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        joblib.dump(self.vectorizer, filepath)

    def load(self, filepath: str = 'models/tfidf_vectorizer.pkl') -> None:
        """Load a saved vectorizer from disk using joblib.

        Args:
            filepath: Path to the serialized vectorizer file. Defaults to 'models/tfidf_vectorizer.pkl'.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer file not found at: {filepath}")

        self.vectorizer = joblib.load(filepath)

    def get_feature_names(self) -> List[str]:
        """Return feature names from the fitted vectorizer.

        Returns:
            List[str]: List of feature name strings.

        Raises:
            ValueError: If the vectorizer has not been fitted or loaded.
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer is not fitted. Call 'fit_transform' or 'load' first.")

        if hasattr(self.vectorizer, "get_feature_names_out"):
            return self.vectorizer.get_feature_names_out().tolist()
        return self.vectorizer.get_feature_names()
