"""LSTM deep learning model module for SMS Spam Detection.

This module provides utilities to build, prepare sequences for, train, and
persist Bidirectional LSTM models using TensorFlow and Keras.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def build_lstm_model(
    vocab_size: int,
    max_length: int,
    embedding_dim: int = 128,
) -> tf.keras.Model:
    """Build and compile a Bidirectional LSTM model for binary classification.

    Architecture:
        Embedding(vocab_size, embedding_dim, input_length=max_length)
        -> Bidirectional(LSTM(64))
        -> Dropout(0.5)
        -> Dense(32, activation='relu')
        -> Dropout(0.3)
        -> Dense(1, activation='sigmoid')

    Args:
        vocab_size: Size of the vocabulary (number of unique tokens).
        max_length: Maximum sequence length.
        embedding_dim: Dimension of the dense embedding. Defaults to 128.

    Returns:
        tf.keras.Model: Compiled TensorFlow Keras model.
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model


def prepare_sequences(
    texts: pd.Series,
    tokenizer: Optional[Tokenizer] = None,
    max_length: int = 200,
    vocab_size: int = 10000,
) -> Tuple[np.ndarray, Tokenizer]:
    """Tokenize and pad text sequences for LSTM input.

    Args:
        texts: Text data to preprocess (pandas Series of strings).
        tokenizer: Existing Tokenizer instance. If None, a new Tokenizer is created
            and fitted on texts. Defaults to None.
        max_length: Maximum length for sequence padding. Defaults to 200.
        vocab_size: Maximum vocabulary size for a new Tokenizer. Defaults to 10000.

    Returns:
        Tuple[np.ndarray, Tokenizer]: Padded sequences as a numpy array and the fitted Tokenizer.
    """
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    return padded_sequences, tokenizer


def train_lstm(
    X_train_text: pd.Series,
    y_train: Any,
    X_val_text: pd.Series,
    y_val: Any,
    epochs: int = 10,
    batch_size: int = 32,
    vocab_size: int = 10000,
    max_length: int = 200,
) -> Tuple[tf.keras.Model, Tokenizer, tf.keras.callbacks.History]:
    """Prepare sequences, build, and train the Bidirectional LSTM model with early stopping.

    Args:
        X_train_text: Training text data (pandas Series).
        y_train: Training labels (array-like).
        X_val_text: Validation text data (pandas Series).
        y_val: Validation labels (array-like).
        epochs: Maximum number of training epochs. Defaults to 10.
        batch_size: Number of samples per gradient update. Defaults to 32.
        vocab_size: Maximum vocabulary size. Defaults to 10000.
        max_length: Maximum length of sequences. Defaults to 200.

    Returns:
        Tuple[tf.keras.Model, Tokenizer, tf.keras.callbacks.History]: Trained model,
        fitted tokenizer, and training history.
    """
    X_train_padded, tokenizer = prepare_sequences(
        texts=X_train_text,
        tokenizer=None,
        max_length=max_length,
        vocab_size=vocab_size,
    )

    X_val_padded, _ = prepare_sequences(
        texts=X_val_text,
        tokenizer=tokenizer,
        max_length=max_length,
        vocab_size=vocab_size,
    )

    model = build_lstm_model(
        vocab_size=vocab_size,
        max_length=max_length,
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_padded,
        np.asarray(y_train),
        validation_data=(X_val_padded, np.asarray(y_val)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
    )

    return model, tokenizer, history


def save_lstm(
    model: tf.keras.Model,
    tokenizer: Tokenizer,
    model_path: str = 'models/lstm_model.keras',
    tokenizer_path: str = 'models/lstm_tokenizer.pkl',
) -> None:
    """Save the trained LSTM model and tokenizer to disk.

    Creates parent directories if they do not exist.

    Args:
        model: Trained Keras model.
        tokenizer: Fitted Tokenizer instance.
        model_path: Destination path for the model. Defaults to 'models/lstm_model.keras'.
        tokenizer_path: Destination path for the tokenizer. Defaults to 'models/lstm_tokenizer.pkl'.
    """
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    model.save(model_path)

    tokenizer_dir = os.path.dirname(tokenizer_path)
    if tokenizer_dir:
        os.makedirs(tokenizer_dir, exist_ok=True)
    joblib.dump(tokenizer, tokenizer_path)


def load_lstm(
    model_path: str = 'models/lstm_model.keras',
    tokenizer_path: str = 'models/lstm_tokenizer.pkl',
) -> Tuple[tf.keras.Model, Tokenizer]:
    """Load a saved LSTM model and tokenizer from disk.

    Args:
        model_path: Path to the serialized Keras model. Defaults to 'models/lstm_model.keras'.
        tokenizer_path: Path to the serialized tokenizer. Defaults to 'models/lstm_tokenizer.pkl'.

    Returns:
        Tuple[tf.keras.Model, Tokenizer]: Loaded Keras model and Tokenizer.

    Raises:
        FileNotFoundError: If either the model or tokenizer file is not found.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at: {tokenizer_path}")

    model = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)

    return model, tokenizer
