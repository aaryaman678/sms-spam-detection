"""Model evaluation and visualization module for SMS Spam Detection.

Provides functions to compute classification metrics, generate confusion matrix
heatmaps, tabulate cross-model comparison results, and plot metric comparisons.
"""

import os
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    model_name: str = "Model",
) -> dict:
    """Evaluate classification model predictions against ground truth labels.

    Computes standard performance metrics (accuracy, precision, recall, and F1 score),
    prints a formatted classification report to stdout, and returns the metrics in a dictionary.

    Args:
        y_true: Ground truth target labels (0 for Ham, 1 for Spam).
        y_pred: Predicted target labels returned by the model.
        model_name: Name of the model being evaluated. Defaults to 'Model'.

    Returns:
        dict: Dictionary containing metric scores with keys:
            - 'model_name': Name of the model
            - 'accuracy': Accuracy score (float)
            - 'precision': Precision score (float)
            - 'recall': Recall score (float)
            - 'f1_score': F1 score (float)
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    print(f"\n{'=' * 20} {model_name} Evaluation Report {'=' * 20}")
    try:
        report = classification_report(
            y_true, y_pred, target_names=["Ham", "Spam"], zero_division=0
        )
    except Exception:
        report = classification_report(y_true, y_pred, zero_division=0)

    print(report)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("=" * (47 + len(model_name)) + "\n")

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def plot_confusion_matrix(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    model_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot and optionally save a confusion matrix heatmap.

    Creates a confusion matrix with `sklearn.metrics.confusion_matrix` and displays
    a heatmap using Seaborn with annotations, formatted as integers, using the 'Blues' colormap.
    Axes are labeled with ['Ham', 'Spam'].

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        model_name: Name of the model for the plot title.
        save_path: Optional file path to save the generated figure. If provided,
            parent directories will be created automatically. If None, displays via plt.show().
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
    )
    plt.title(f"{model_name} - Confusion Matrix", fontsize=14, pad=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Aggregate, sort, and print model comparison results.

    Accepts a list of dictionaries as returned by `evaluate_model`, creates
    a pandas DataFrame, sorts rows in descending order by `f1_score`, prints
    a formatted comparison table, and returns the DataFrame.

    Args:
        results: List of result dictionaries from `evaluate_model`.

    Returns:
        pd.DataFrame: Formatted comparison DataFrame sorted by `f1_score` descending.
    """
    results_df = pd.DataFrame(results)
    if "f1_score" in results_df.columns:
        results_df = results_df.sort_values(by="f1_score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 30 + " Model Comparison Table " + "=" * 30)
    print(results_df.to_string(index=False))
    print("=" * 84 + "\n")

    return results_df


def plot_model_comparison(results_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot and optionally save a grouped bar chart comparing performance metrics across models.

    Creates a grouped bar chart comparing accuracy, precision, recall, and F1 score across models
    using Seaborn and Matplotlib.

    Args:
        results_df: DataFrame containing model evaluation metrics as produced by
            `compare_models`.
        save_path: Optional file path to save the generated figure. If provided,
            parent directories will be created automatically. If None, displays via plt.show().
    """
    if results_df.empty:
        print("Warning: results_df is empty. Skipping model comparison plot.")
        return

    metric_cols = ["accuracy", "precision", "recall", "f1_score"]
    available_metrics = [col for col in metric_cols if col in results_df.columns]

    # Melt DataFrame for Seaborn grouped bar plot
    melted_df = results_df.melt(
        id_vars=["model_name"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score",
    )

    label_mapping = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
    }
    melted_df["Metric"] = melted_df["Metric"].map(label_mapping).fillna(melted_df["Metric"])

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(
        data=melted_df,
        x="model_name",
        y="Score",
        hue="Metric",
        palette="Blues_r",
    )

    plt.title("Model Performance Comparison", fontsize=15, pad=15)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.legend(title="Metric", loc="lower right")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Model comparison plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()
