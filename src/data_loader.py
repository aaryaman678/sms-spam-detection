"""Data loader module for the SMS Spam Detection project.

Provides functions to download the UCI SMS Spam Collection dataset,
load and encode labels, and perform train/test splits.
"""

from pathlib import Path
from typing import Tuple, Union
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split


def download_dataset(data_dir: Union[str, Path]) -> Path:
    """Download and extract the UCI SMS Spam Collection dataset.

    Downloads the dataset zip archive from the UCI Machine Learning Repository,
    extracts the raw tab-separated file, reads it, and saves it as a CSV file.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Directory path where the dataset should be downloaded and saved.

    Returns
    -------
    Path
        Path to the saved 'spam.csv' file.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = data_path / "sms_spam_collection.zip"
    csv_path = data_path / "spam.csv"

    # Download archive
    urllib.request.urlretrieve(url, zip_path)

    # Extract archive
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    # Locate raw file
    raw_file = data_path / "SMSSpamCollection"
    if not raw_file.exists():
        matches = list(data_path.glob("**/SMSSpamCollection*"))
        if matches:
            raw_file = matches[0]
        else:
            raise FileNotFoundError(
                f"SMSSpamCollection file not found in {data_path} after extraction."
            )

    # Read tab-separated file (columns: label, message)
    try:
        df = pd.read_csv(
            raw_file,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="utf-8",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            raw_file,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="latin-1",
        )

    # Save as CSV
    df.to_csv(csv_path, index=False)
    return csv_path


def load_data(filepath: Union[str, Path] = "data/spam.csv") -> pd.DataFrame:
    """Load the SMS Spam dataset, downloading if necessary, and encode labels.

    Parameters
    ----------
    filepath : Union[str, Path], default='data/spam.csv'
        Path to the dataset CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 'label', 'message', and 'label_encoded' columns.
    """
    path = Path(filepath)
    if not path.exists():
        download_dataset(path.parent)

    df = pd.read_csv(path)

    # Encode labels: ham=0, spam=1
    df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})

    # Print class distribution
    print("Class distribution:")
    print(df["label"].value_counts())
    print("\nClass distribution (proportions):")
    print(df["label_encoded"].value_counts(normalize=True))

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Perform stratified train/test split on messages and encoded labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'message' and 'label_encoded' columns.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random state seed for reproducibility.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    X = df["message"]
    y = df["label_encoded"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
