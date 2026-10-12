"""Interactive Streamlit Web Application for SMS Spam Detection.

A production-grade, presentation-ready academic demonstration showcasing:
- Real-time classification with multiple ML models
- Live side-by-side model comparison
- Explainable NLP keyword attribution
- Interactive step-by-step NLP pipeline inspector
- Benchmark evaluation dashboards & confusion matrices
- Exploratory Data Analysis (EDA) gallery

Usage:
    streamlit run app.py
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.predict import explain_keywords, predict_all_models, predict_sms
from src.preprocessor import clean_text, preprocess_text

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Detector — NLP Project",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom presentation-grade CSS styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Top Header & Summary Metrics Banner
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🛡️ SMS Spam Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Natural Language Processing Subject Project • Text Preprocessing, TF-IDF Vectorization & Machine Learning</div>',
    unsafe_allow_html=True,
)

# Top metric summary cards
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">5,574</div>
            <div class="metric-label">UCI Dataset SMS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value" style="color: #059669;">98.48%</div>
            <div class="metric-label">Best Accuracy (Linear SVM)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value" style="color: #2563EB;">100.0%</div>
            <div class="metric-label">Naive Bayes Precision (0 FP)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_m4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">5,000</div>
            <div class="metric-label">TF-IDF N-Gram Features</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Project Controls")

    default_model = st.selectbox(
        "Active Classifier:",
        options=["svm", "naive_bayes", "logistic_regression"],
        format_func=lambda x: {
            "svm": "Linear SVM (Highest Accuracy)",
            "naive_bayes": "Multinomial Naive Bayes (Zero False Positives)",
            "logistic_regression": "Logistic Regression (Calibrated Probabilities)",
        }[x],
        index=0,
    )

    st.markdown("---")
    st.subheader("📚 NLP Pipeline Steps")
    st.markdown(
        """
        1. **Noise Cleaning**: Regex strips URLs, emails, phone numbers & punctuation.
        2. **Case Normalization**: All characters to lowercase.
        3. **Tokenization**: NLTK `word_tokenize`.
        4. **Stopword Filtering**: English stopwords removed.
        5. **Lemmatization**: WordNet lemmatizer extracts root lemmas.
        6. **Vectorization**: TF-IDF (1-2 grams, max 5,000 features).
        7. **Inference**: High-dimensional hyperplane separation.
        """
    )

    st.markdown("---")
    st.caption("SMS Spam Detection • NLP Subject Project • 2026")

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_live, tab_pipeline, tab_benchmarks, tab_eda = st.tabs(
    [
        "🎯 Live SMS Detector",
        "🔬 NLP Pipeline Inspector",
        "📊 Model Benchmarks & Metrics",
        "📈 Dataset & EDA Visualizations",
    ]
)

# =============================================================================
# TAB 1: Live SMS Detector & Comparison
# =============================================================================
with tab_live:
    st.subheader("Real-time Message Classification")

    # Session state for input
    if "sms_text" not in st.session_state:
        st.session_state.sms_text = ""

    def set_example(text: str):
        st.session_state.sms_text = text

    # Curated Preset Scenarios for demonstration
    st.markdown("**⚡ Quick Test Presets (Click to Load):**")
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown("🚨 *Spam & Smishing Scenarios:*")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🎰 Lottery Scam", use_container_width=True):
                set_example("URGENT! You have won a 1,000 cash prize or a holiday! Call 09050000327 to claim NOW. T&Cs apply.")
        with c2:
            if st.button("💳 Bank Phishing", use_container_width=True):
                set_example("Dear customer, your bank account is suspended due to suspicious activity. Verify immediately at http://bit.ly/bank-sec to prevent permanent block.")
        with c3:
            if st.button("📱 iPhone Winner", use_container_width=True):
                set_example("Congratulations! You have been selected to win a free iPhone 15 Pro. Reply CLAIM to 88099 now. Offer expires in 2 hours.")

    with p_col2:
        st.markdown("💬 *Legitimate (Ham) Scenarios:*")
        c4, c5, c6 = st.columns(3)
        with c4:
            if st.button("👥 Meeting Friend", use_container_width=True):
                set_example("Hey, are we still meeting for lunch today at 12:30? Let me know if you are running late.")
        with c5:
            if st.button("🔐 2FA / OTP Code", use_container_width=True):
                set_example("Your verification security code is 482910. Valid for 5 minutes. Do not share this OTP with anyone.")
        with c6:
            if st.button("📦 Delivery Update", use_container_width=True):
                set_example("Your package with tracking number 94001092023 will be delivered by 5 PM today. Thank you for shopping with us.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Input message box
    user_sms = st.text_area(
        "Enter or edit SMS text message:",
        value=st.session_state.sms_text,
        height=100,
        placeholder="Type or paste any SMS text message here to evaluate...",
    )

    compare_all = st.checkbox("🔍 Compare all 3 models side-by-side simultaneously", value=True)

    if st.button("⚡ Run Spam Detection", type="primary", use_container_width=True):
        if not user_sms.strip():
            st.warning("Please enter an SMS message or click one of the quick test presets above.")
        else:
            st.markdown("---")
            if compare_all:
                st.markdown("### 🏆 Side-by-Side Model Comparison")
                all_preds = predict_all_models(user_sms)

                cols = st.columns(3)
                model_titles = {
                    "svm": ("Linear SVM", "Best Accuracy (98.5%)"),
                    "naive_bayes": ("Naive Bayes", "Zero False Positives (100% Prec.)"),
                    "logistic_regression": ("Logistic Regression", "Probabilistic Calibration"),
                }

                for idx, (m_key, (m_name, m_desc)) in enumerate(model_titles.items()):
                    with cols[idx]:
                        res = all_preds.get(m_key, {})
                        pred_label = res.get("prediction", "unknown").upper()
                        conf = res.get("confidence")

                        if pred_label == "SPAM":
                            st.error(f"### 🚨 {pred_label}\n**{m_name}**\n*{m_desc}*")
                        else:
                            st.success(f"### ✅ {pred_label}\n**{m_name}**\n*{m_desc}*")

                        if conf is not None:
                            st.progress(conf, text=f"Confidence: {conf*100:.1f}%")
                        else:
                            st.caption("Confidence score: High margin separation")

            else:
                # Single model inference
                single_res = predict_sms(user_sms, model_type=default_model)
                pred_label = single_res["prediction"].upper()
                conf = single_res["confidence"]

                res_col1, res_col2 = st.columns([1, 1])
                with res_col1:
                    if pred_label == "SPAM":
                        st.error(f"## 🚨 SPAM DETECTED\n**Model:** {default_model.upper()}")
                    else:
                        st.success(f"## ✅ LEGITIMATE MESSAGE (HAM)\n**Model:** {default_model.upper()}")

                with res_col2:
                    if conf is not None:
                        st.metric("Model Confidence", f"{conf * 100:.2f}%")
                        st.progress(conf)

            # -------------------------------------------------------------
            # Feature Attribution / Explainability Section
            # -------------------------------------------------------------
            st.markdown("### 🧠 Model Explainability & Keyword Attribution")
            st.markdown(
                "Why did the model classify this message? The table below highlights the tokens that most strongly influenced the classification based on learned feature weights:"
            )

            keywords = explain_keywords(user_sms, top_n=8)
            if keywords:
                exp_col1, exp_col2 = st.columns([1, 1])

                with exp_col1:
                    df_keywords = pd.DataFrame(keywords)[["word", "impact", "weight"]]
                    df_keywords.columns = ["Vocabulary Token", "Classification Impact", "Learned Weight"]
                    st.dataframe(df_keywords, use_container_width=True, hide_index=True)

                with exp_col2:
                    # Bar chart of top contributing keywords
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    words = [k["word"] for k in reversed(keywords)]
                    weights = [k["weight"] for k in reversed(keywords)]
                    colors = ["#EF4444" if w > 0 else "#10B981" for w in weights]

                    ax.barh(words, weights, color=colors)
                    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
                    ax.set_title("Feature Contribution Weights (Red=Spam, Green=Ham)", fontsize=11)
                    ax.set_xlabel("Logistic Regression Coefficient", fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("No specific vocabulary tokens from the 5,000-word feature space were strongly weighted in this message.")

# =============================================================================
# TAB 2: NLP Pipeline Inspector (Step-by-Step)
# =============================================================================
with tab_pipeline:
    st.subheader("Step-by-Step NLP Transformation Pipeline")
    st.markdown(
        "Demonstrate to your professor how raw text flows through each Natural Language Processing stage before reaching the vectorizer:"
    )

    test_sentence = st.text_input(
        "Input text to inspect through the NLP pipeline:",
        value=user_sms if user_sms else "URGENT! Call 09050000327 to claim your 1,000 cash prize now at http://bit.ly/prize!",
    )

    if test_sentence:
        # Step 1: Raw
        st.markdown("#### 1. Raw Input")
        st.code(test_sentence, language="text")

        # Step 2: Regex noise cleaning
        cleaned = clean_text(test_sentence)
        st.markdown("#### 2. Regex Cleaning & Noise Removal")
        st.caption("Strips URLs, email addresses, phone digits, and special characters; converts to lowercase.")
        st.code(cleaned, language="text")

        # Step 3: Tokenization
        import nltk
        tokens = nltk.word_tokenize(cleaned)
        st.markdown("#### 3. Tokenization (`nltk.word_tokenize`)")
        st.caption(f"Splits text into individual linguistic tokens ({len(tokens)} total tokens).")
        st.write(tokens)

        # Step 4: Stopwords removal & Lemmatization
        final_preprocessed = preprocess_text(test_sentence)
        st.markdown("#### 4. Stopword Filtering & Lemmatization (`WordNetLemmatizer`)")
        st.caption("Filters out high-frequency functional words ('is', 'to', 'your') and reduces tokens to root lemmas.")
        st.code(final_preprocessed, language="text")

        # Step 5: TF-IDF Vector representation
        import joblib
        vec_path = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.pkl")
        if os.path.exists(vec_path):
            vec = joblib.load(vec_path)
            tfidf_vec = vec.transform([final_preprocessed])
            non_zero_indices = tfidf_vec.nonzero()[1]
            feature_names = np.array(vec.get_feature_names_out())

            st.markdown("#### 5. TF-IDF Vector Space Representation")
            st.caption(f"Mapped into 5,000-dimensional sparse feature space ({len(non_zero_indices)} active features).")

            if len(non_zero_indices) > 0:
                active_features = [
                    {"Feature (N-Gram)": feature_names[idx], "TF-IDF Score": round(float(tfidf_vec[0, idx]), 4)}
                    for idx in non_zero_indices
                ]
                st.dataframe(pd.DataFrame(active_features), use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3: Model Benchmarks & Metrics
# =============================================================================
with tab_benchmarks:
    st.subheader("Model Benchmark Evaluation on Held-Out Test Set")
    st.markdown(
        "Evaluated on **1,115 test samples** with stratified 80/20 train-test partitioning (966 legitimate, 149 spam):"
    )

    metrics_df = pd.DataFrame(
        [
            {"Model": "Linear SVM", "Accuracy": "98.48%", "Precision": "97.83%", "Recall": "90.60%", "F1-Score": "0.9408", "Rank": "🥇 Best Overall"},
            {"Model": "Naive Bayes (MultinomialNB)", "Accuracy": "97.04%", "Precision": "100.00%", "Recall": "77.85%", "F1-Score": "0.8755", "Rank": "🥈 0 False Positives"},
            {"Model": "Logistic Regression", "Accuracy": "97.04%", "Precision": "100.00%", "Recall": "77.85%", "F1-Score": "0.8755", "Rank": "🥉 Calibrated"},
        ]
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Key Academic Insight Box
    st.info(
        """
        **💡 Key NLP & Machine Learning Insights for Viva / Presentation:**
        - **Why Precision Matters Most**: In spam filtering, marking a legitimate message (e.g., job offer, doctor appointment, OTP) as spam is far more disruptive than letting a spam message slip through. Naive Bayes achieved **100% precision (0 false positives)** on the test set.
        - **Why Linear SVM Won on F1**: Linear SVM constructs a maximum-margin hyperplane in high-dimensional TF-IDF space, achieving significantly higher recall (**90.60% vs 77.85%**) while maintaining 97.8% precision.
        """
    )

    st.markdown("---")
    st.subheader("Model Evaluation Visualizations")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        comp_img = os.path.join(PROJECT_ROOT, "models", "model_comparison.png")
        if os.path.exists(comp_img):
            st.image(comp_img, caption="Grouped Performance Metric Comparison", use_container_width=True)

    with col_chart2:
        svm_cm = os.path.join(PROJECT_ROOT, "models", "svm_confusion_matrix.png")
        if os.path.exists(svm_cm):
            st.image(svm_cm, caption="Linear SVM Confusion Matrix Heatmap", use_container_width=True)

    st.markdown("#### Individual Model Confusion Matrices")
    cm_c1, cm_c2 = st.columns(2)
    with cm_c1:
        nb_cm = os.path.join(PROJECT_ROOT, "models", "naive_bayes_confusion_matrix.png")
        if os.path.exists(nb_cm):
            st.image(nb_cm, caption="Naive Bayes Confusion Matrix (0 False Positives)", use_container_width=True)
    with cm_c2:
        lr_cm = os.path.join(PROJECT_ROOT, "models", "logistic_regression_confusion_matrix.png")
        if os.path.exists(lr_cm):
            st.image(lr_cm, caption="Logistic Regression Confusion Matrix", use_container_width=True)

# =============================================================================
# TAB 4: Dataset & EDA Visualizations
# =============================================================================
with tab_eda:
    st.subheader("Exploratory Data Analysis (EDA) Gallery")
    st.markdown(
        "Comprehensive visual exploration of the **UCI SMS Spam Collection** (5,574 messages):"
    )

    eda_dir = os.path.join(PROJECT_ROOT, "notebooks", "figures")

    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        cd_img = os.path.join(eda_dir, "class_distribution.png")
        if os.path.exists(cd_img):
            st.image(cd_img, caption="Class Distribution: 86.6% Ham vs 13.4% Spam", use_container_width=True)

    with eda_col2:
        pie_img = os.path.join(eda_dir, "spam_percentage_pie.png")
        if os.path.exists(pie_img):
            st.image(pie_img, caption="Dataset Class Proportions (Imbalanced Classification)", use_container_width=True)

    st.markdown("---")
    st.subheader("Message Length & Word Count Distributions")
    len_c1, len_c2 = st.columns(2)
    with len_c1:
        len_img = os.path.join(eda_dir, "message_length_distribution.png")
        if os.path.exists(len_img):
            st.image(len_img, caption="Character Count Distribution (Spam messages peak near 140-160 chars)", use_container_width=True)
    with len_c2:
        wc_img = os.path.join(eda_dir, "word_count_distribution.png")
        if os.path.exists(wc_img):
            st.image(wc_img, caption="Word Count Distribution by Class", use_container_width=True)

    st.markdown("---")
    st.subheader("Word Clouds & Keyword Frequencies")
    wc_both = os.path.join(eda_dir, "wordclouds_side_by_side.png")
    if os.path.exists(wc_both):
        st.image(wc_both, caption="Side-by-Side Word Clouds: Spam (Urgency/Monetary) vs Ham (Conversational)", use_container_width=True)

    top_words = os.path.join(eda_dir, "top_words.png")
    if os.path.exists(top_words):
        st.image(top_words, caption="Top 20 Most Frequent Words: Spam vs Ham", use_container_width=True)
