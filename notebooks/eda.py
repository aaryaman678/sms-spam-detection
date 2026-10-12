"""
Exploratory Data Analysis (EDA) Script for SMS Spam Detection.

Generates comprehensive data visualizations and statistical summaries for the
SMS Spam dataset, including class balance distributions, character and word count
distributions, word clouds, top n-gram frequency charts, and pie charts.
All visual artifacts are automatically saved into the notebooks/figures directory.
"""

import os
import sys
from collections import Counter
from typing import List, Tuple, Union

# Ensure the parent project root directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src import data_loader, preprocessor

# Set matplotlib style with backward-compatible fallbacks
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("default")

# Configure figures output directory relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SAVE_DIR = os.path.join(BASE_DIR, "notebooks", "figures")
os.makedirs(SAVE_DIR, exist_ok=True)


def _extract_tokens(series: pd.Series) -> List[str]:
    """Extract a flattened list of tokens from a series of preprocessed messages.

    Args:
        series: Pandas series containing preprocessed text or lists of tokens.

    Returns:
        Flattened list of string tokens in lowercase.
    """
    tokens: List[str] = []
    for item in series:
        if isinstance(item, (list, tuple)):
            tokens.extend([str(t).lower() for t in item if str(t).strip()])
        elif isinstance(item, str):
            tokens.extend([w.lower() for w in item.split() if w.strip()])
    return tokens


def run_eda() -> None:
    """Execute complete EDA pipeline and save all visual plots to disk."""
    print("Loading SMS dataset for exploratory analysis...")
    df = data_loader.load_data()

    # Identify and standardize column names
    label_col = "label" if "label" in df.columns else ("v1" if "v1" in df.columns else df.columns[0])
    text_col = "message" if "message" in df.columns else ("text" if "text" in df.columns else ("v2" if "v2" in df.columns else df.columns[1]))

    # Normalize labels to lowercase ('ham' and 'spam')
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    total_messages = len(df)

    color_palette = {"ham": "#2ecc71", "spam": "#e74c3c"}

    # =========================================================================
    # 1. Class Distribution Bar Chart
    # =========================================================================
    print("Generating class distribution chart...")
    fig, ax = plt.subplots(figsize=(7, 5))
    class_counts = df[label_col].value_counts()
    bar_colors = [color_palette.get(cls, "#3498db") for cls in class_counts.index]

    bars = ax.bar(
        class_counts.index.str.capitalize(),
        class_counts.values,
        color=bar_colors,
        edgecolor="black",
        linewidth=1,
        width=0.5,
    )

    for bar in bars:
        height = bar.get_height()
        pct = (height / total_messages) * 100
        ax.annotate(
            f"{int(height):,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title("SMS Class Distribution (Ham vs. Spam)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Message Class", fontsize=12)
    ax.set_ylabel("Message Count", fontsize=12)
    ax.set_ylim(0, max(class_counts.values) * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "class_distribution.png"), dpi=300)
    plt.close(fig)

    # =========================================================================
    # 2. Message Length Distribution (Character Count)
    # =========================================================================
    print("Generating message length distribution...")
    df["msg_length"] = df[text_col].astype(str).str.len()

    fig, ax = plt.subplots(figsize=(9, 5))
    ham_lengths = df[df[label_col] == "ham"]["msg_length"]
    spam_lengths = df[df[label_col] == "spam"]["msg_length"]

    sns.histplot(
        ham_lengths,
        color=color_palette["ham"],
        label="Ham (Legitimate)",
        kde=True,
        alpha=0.5,
        bins=50,
        ax=ax,
    )
    sns.histplot(
        spam_lengths,
        color=color_palette["spam"],
        label="Spam",
        kde=True,
        alpha=0.5,
        bins=50,
        ax=ax,
    )

    ax.set_title("Message Character Length Distribution by Class", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Character Count", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(title="Message Type", loc="upper right")
    ax.set_xlim(0, max(df["msg_length"].quantile(0.995) * 1.1, 250))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "message_length_distribution.png"), dpi=300)
    plt.close(fig)

    # =========================================================================
    # 3. Word Count Distribution
    # =========================================================================
    print("Generating word count distribution...")
    df["word_count"] = df[text_col].astype(str).apply(lambda text: len(text.split()))

    fig, ax = plt.subplots(figsize=(9, 5))
    ham_words = df[df[label_col] == "ham"]["word_count"]
    spam_words = df[df[label_col] == "spam"]["word_count"]

    sns.histplot(
        ham_words,
        color=color_palette["ham"],
        label="Ham (Legitimate)",
        kde=True,
        alpha=0.5,
        bins=40,
        ax=ax,
    )
    sns.histplot(
        spam_words,
        color=color_palette["spam"],
        label="Spam",
        kde=True,
        alpha=0.5,
        bins=40,
        ax=ax,
    )

    ax.set_title("Word Count Distribution by Class", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Word Count", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(title="Message Type", loc="upper right")
    ax.set_xlim(0, max(df["word_count"].quantile(0.995) * 1.1, 50))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "word_count_distribution.png"), dpi=300)
    plt.close(fig)

    # =========================================================================
    # 4. Word Clouds (Spam and Ham)
    # =========================================================================
    print("Preprocessing messages for word clouds and n-gram analysis...")
    preprocessed_series = preprocessor.preprocess_series(df[text_col])

    # Convert preprocessed tokens into consolidated string for each row
    def stringify_cleaned(entry: Union[str, List[str]]) -> str:
        if isinstance(entry, (list, tuple)):
            return " ".join(str(w) for w in entry)
        return str(entry)

    df["cleaned_text"] = preprocessed_series.apply(stringify_cleaned)

    spam_clean_corpus = " ".join(df[df[label_col] == "spam"]["cleaned_text"]).strip()
    ham_clean_corpus = " ".join(df[df[label_col] == "ham"]["cleaned_text"]).strip()

    wc_spam = WordCloud(
        colormap="Reds",
        background_color="white",
        max_words=100,
        width=800,
        height=500,
        random_state=42,
    ).generate(spam_clean_corpus if spam_clean_corpus else "spam empty")

    wc_ham = WordCloud(
        colormap="Greens",
        background_color="white",
        max_words=100,
        width=800,
        height=500,
        random_state=42,
    ).generate(ham_clean_corpus if ham_clean_corpus else "ham empty")

    # Save individual spam wordcloud
    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    ax_s.imshow(wc_spam, interpolation="bilinear")
    ax_s.axis("off")
    ax_s.set_title("Most Frequent Words in Spam Messages", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "wordcloud_spam.png"), dpi=300)
    plt.close(fig_s)

    # Save individual ham wordcloud
    fig_h, ax_h = plt.subplots(figsize=(8, 5))
    ax_h.imshow(wc_ham, interpolation="bilinear")
    ax_h.axis("off")
    ax_h.set_title("Most Frequent Words in Ham Messages", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "wordcloud_ham.png"), dpi=300)
    plt.close(fig_h)

    # Save side-by-side figure
    fig_clouds, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(16, 6))
    ax_c1.imshow(wc_spam, interpolation="bilinear")
    ax_c1.set_title("Spam Word Cloud", fontsize=15, fontweight="bold")
    ax_c1.axis("off")

    ax_c2.imshow(wc_ham, interpolation="bilinear")
    ax_c2.set_title("Ham Word Cloud", fontsize=15, fontweight="bold")
    ax_c2.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "wordclouds_side_by_side.png"), dpi=300)
    plt.close(fig_clouds)

    # =========================================================================
    # 5. Top 20 Words Bar Charts Side by Side
    # =========================================================================
    print("Calculating top 20 words for spam and ham...")
    spam_tokens = _extract_tokens(df[df[label_col] == "spam"]["cleaned_text"])
    ham_tokens = _extract_tokens(df[df[label_col] == "ham"]["cleaned_text"])

    spam_top20 = Counter(spam_tokens).most_common(20)
    ham_top20 = Counter(ham_tokens).most_common(20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    if spam_top20:
        s_words, s_counts = zip(*reversed(spam_top20))
        ax1.barh(s_words, s_counts, color=color_palette["spam"], edgecolor="black", alpha=0.85)
        ax1.set_title("Top 20 Words in Spam Messages", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Term Frequency", fontsize=12)
        ax1.grid(axis="x", linestyle="--", alpha=0.7)

    if ham_top20:
        h_words, h_counts = zip(*reversed(ham_top20))
        ax2.barh(h_words, h_counts, color=color_palette["ham"], edgecolor="black", alpha=0.85)
        ax2.set_title("Top 20 Words in Ham Messages", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Term Frequency", fontsize=12)
        ax2.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "top_words.png"), dpi=300)
    plt.close(fig)

    # =========================================================================
    # 6. Spam Percentage Pie Chart
    # =========================================================================
    print("Generating spam percentage pie chart...")
    fig, ax = plt.subplots(figsize=(6, 6))
    pie_counts = df[label_col].value_counts()
    pie_colors = [color_palette.get(c, "#3498db") for c in pie_counts.index]
    pie_labels = [c.capitalize() for c in pie_counts.index]
    explode = [0.08 if c == "spam" else 0.0 for c in pie_counts.index]

    ax.pie(
        pie_counts.values,
        labels=pie_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=pie_colors,
        explode=explode,
        shadow=True,
        textprops={"fontsize": 12, "fontweight": "bold"},
    )
    ax.set_title("Spam vs. Ham Proportion", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "spam_percentage_pie.png"), dpi=300)
    plt.close(fig)

    print("EDA complete! Figures saved to notebooks/figures/")


if __name__ == "__main__":
    run_eda()
