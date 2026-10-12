"""Text preprocessing module for the SMS Spam Detection project.

Provides functions to clean raw text, tokenize, remove stopwords,
and lemmatize tokens for machine learning models.
"""

import re
from typing import Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd

# Download required NLTK data resources silently on import
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS: Set[str] = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Clean raw text string according to standard NLP preprocessing steps.

    Performs the following steps:
    1. Converts text to lowercase
    2. Removes URLs (http/https/www patterns)
    3. Removes email addresses
    4. Removes phone numbers (digits with optional dashes/spaces/parens)
    5. Removes all non-alphanumeric characters except spaces
    6. Removes extra whitespace
    7. Returns cleaned string

    Parameters
    ----------
    text : str
        The raw input text string to clean.

    Returns
    -------
    str
        The cleaned text string.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove URLs (http/https/www patterns)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", text)

    # 4. Remove phone numbers (digits with optional dashes/spaces/parens)
    text = re.sub(
        r"(\+?\d{1,3}[\s-]?)?\(?\d{2,5}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}|\b\d{7,15}\b",
        "",
        text,
    )

    # 5. Remove all non-alphanumeric characters except spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # 6. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Return cleaned string
    return text


def preprocess_text(text: str) -> str:
    """Preprocess text by cleaning, tokenizing, removing stopwords, and lemmatizing.

    Performs the following steps:
    1. Calls clean_text
    2. Tokenizes with nltk.word_tokenize
    3. Removes English stopwords
    4. Lemmatizes each token with WordNetLemmatizer
    5. Joins tokens back with spaces and returns

    Parameters
    ----------
    text : str
        The input text string to preprocess.

    Returns
    -------
    str
        The preprocessed text string.
    """
    # 1. Call clean_text
    cleaned = clean_text(text)

    # 2. Tokenize with nltk.word_tokenize
    tokens = word_tokenize(cleaned)

    # 3. Remove English stopwords & 4. Lemmatize each token with WordNetLemmatizer
    processed_tokens = [
        _LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in _STOPWORDS
    ]

    # 5. Join tokens back with spaces and return
    return " ".join(processed_tokens)


def preprocess_series(series: pd.Series) -> pd.Series:
    """Apply text preprocessing to every element in a pandas Series.

    Parameters
    ----------
    series : pd.Series
        Pandas Series containing raw text messages.

    Returns
    -------
    pd.Series
        Pandas Series containing preprocessed text messages.
    """
    return series.apply(preprocess_text)
