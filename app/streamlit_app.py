try:
    import streamlit as st
    from joblib import load
    import os
    import sys
    import json

    # Ensure repository root is on sys.path so `from src...` imports work on Streamlit Cloud
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from src.data_loader import load_data

except Exception:
    # Capture full traceback to files for easier debugging on Streamlit Cloud
    import traceback

    tb = traceback.format_exc()
    try:
        with open("/tmp/streamlit_import_error.log", "w", encoding="utf-8") as f:
            f.write(tb)
    except Exception:
        pass
    try:
        # write next to the app file (may be visible in deploy logs)
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(root, "streamlit_import_error.log"), "w", encoding="utf-8") as f:
            f.write(tb)
    except Exception:
        pass
    # Re-raise so Streamlit shows the error and logs capture the details
    raise

MODEL_PATH = os.path.join("models", "model--logistic--v0.1.joblib")
METRICS_PATH = os.path.join("models", "metrics.json")
CONF_MATRIX_PATH = os.path.join("models", "confusion_matrix.png")
ROC_PATH = os.path.join("models", "roc_curve.png")
METADATA_PATH = os.path.join("models", "metadata.json")


@st.cache_data
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    try:
        return load(path)
    except Exception:
        return None


def load_metrics(path=METRICS_PATH):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_metadata(path=METADATA_PATH):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def show_image(path: str, caption: str = None):
    if os.path.exists(path):
        st.image(path, caption=caption)
    else:
        st.info(f"{caption or 'Image'} not available. Run training to generate it.")


def plot_confusion_matrix_ax(y_true, y_pred, ax=None):
    """Draw a confusion matrix on the given axis (imports locally)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    if ax is None:
        fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    return ax


def plot_roc_ax(estimator, X, y, ax=None):
    """Draw ROC curve using estimator on provided data; handles failures gracefully."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    if ax is None:
        fig, ax = plt.subplots()
    try:
        RocCurveDisplay.from_estimator(estimator, X, y, ax=ax)
    except Exception as e:
        ax.text(0.5, 0.5, f"ROC not available: {e}", ha="center")
    return ax


def plot_precision_recall_ax(estimator, X, y, ax=None):
    """Plot precision-recall curve using estimator and data."""
    try:
        from sklearn.metrics import PrecisionRecallDisplay
    except Exception:
        # Older sklearn fallback
        from sklearn.metrics import precision_recall_curve

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    try:
        # Prefer high-level API
        try:
            PrecisionRecallDisplay.from_estimator(estimator, X, y, ax=ax)
        except Exception:
            # fallback to manual curve
            precisions, recalls, _ = precision_recall_curve(y, estimator.predict_proba(X)[:, 1])
            ax.plot(recalls, precisions, label="Precision-Recall")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
    except Exception as e:
        ax.text(0.5, 0.5, f"PR not available: {e}", ha="center")
    return ax


def plot_threshold_sweep_ax(probs, y, ax=None):
    """Compute precision/recall/f1 over thresholds and plot them."""
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    try:
        thresholds = np.linspace(0.0, 1.0, 101)
        precisions = []
        recalls = []
        f1s = []
        for t in thresholds:
            y_pred_t = (probs >= t).astype(int)
            precisions.append(precision_score(y, y_pred_t, zero_division=0))
            recalls.append(recall_score(y, y_pred_t, zero_division=0))
            f1s.append(f1_score(y, y_pred_t, zero_division=0))

        ax.plot(thresholds, precisions, label="Precision", color="#1f77b4")
        ax.plot(thresholds, recalls, label="Recall", color="#ff7f0e")
        ax.plot(thresholds, f1s, label="F1", color="#2ca02c")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best")
    except Exception as e:
        ax.text(0.5, 0.5, f"threshold sweep failed: {e}", ha="center")
    return ax


def main():
    """Render the Streamlit dashboard."""
    # Local imports to keep module import fast and avoid top-level heavy deps
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        ConfusionMatrixDisplay,
        RocCurveDisplay,
    )

    st.set_page_config(page_title="Spam/Ham Classifier", layout="wide")
    # Sidebar controls
    st.sidebar.title("Controls")
    threshold = st.sidebar.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.01)
    sample_size = st.sidebar.slider("Dataset sample size (for preview)", 1, 50, 10)
    reload_model = st.sidebar.button("Reload model")
    show_prob_hist = st.sidebar.checkbox("Show probability histogram", True)
    show_raw_metrics = st.sidebar.checkbox("Show saved metrics.json", True)

    st.title("Spam/Ham Classifier")

    # Reload cache if requested
    if reload_model:
        try:
            st.experimental_memo_clear()
        except Exception:
            pass

    model = load_model()
    metrics = load_metrics()
    metadata = load_metadata()

    # Layout: stacked sections top -> bottom
    # 1) Predict a message
    st.header("Predict a message")

    # Prepare session state key for the input so buttons can programmatically set it
    if "predict_text" not in st.session_state:
        st.session_state["predict_text"] = "Free entry: claim your prize now"

    # Try to load dataset examples (non-blocking)
    try:
        df_examples = load_data(download=False)
    except Exception:
        df_examples = None

    # Determine example messages (clean sampled text to avoid garbled outputs)
    spam_example = "Free entry: claim your prize now"
    ham_example = "Hey, are we still on for tonight?"

    def _clean_example_text(x: "any") -> str:
        """Normalize and clean a candidate example message.

        Returns empty string if the candidate is unsuitable.
        """
        try:
            # handle bytes
            if isinstance(x, (bytes, bytearray)):
                x = x.decode("utf-8", errors="replace")
            s = str(x)
            # lightweight unicode normalization and control-char removal
            import unicodedata, re

            s = unicodedata.normalize("NFKC", s)
            # remove C0/C1 control chars
            s = re.sub(r"[\x00-\x1f\x7f-\x9f]+", " ", s)
            # collapse whitespace
            s = re.sub(r"\s+", " ", s).strip()
            # require some minimal length and at least one alphanumeric or CJK char
            if len(s) < 8:
                return ""
            if re.search(r"[A-Za-z0-9\u4e00-\u9fff]", s) is None:
                return ""
            # ensure ends with punctuation for nicer display
            if s and s[-1] not in ".!?。！？":
                s = s + "."
            return s
        except Exception:
            return ""

    try:
        if df_examples is not None:
            s = df_examples[df_examples["label"].str.lower() == "spam"]
            h = df_examples[df_examples["label"].str.lower() == "ham"]

            if not s.empty:
                # prefer a cleaned, reasonably long example; fall back to longest cleaned
                candidates = s["message"].astype(object).apply(_clean_example_text)
                candidates = candidates[candidates != ""]
                if not candidates.empty:
                    # choose the longest cleaned example for readability
                    spam_example = candidates.loc[candidates.str.len().idxmax()]

            if not h.empty:
                candidates = h["message"].astype(object).apply(_clean_example_text)
                candidates = candidates[candidates != ""]
                if not candidates.empty:
                    # Prefer a "complete sentence" candidate for ham examples.
                    # Define complete sentence as: at least min_chars and at least min_words,
                    # and (we already ensured a trailing punctuation in _clean_example_text).
                    min_chars = 20
                    min_words = 3
                    def _is_full_sentence(s: str) -> bool:
                        if not s:
                            return False
                        if len(s) < min_chars:
                            return False
                        # count words (split on whitespace)
                        if len(s.split()) < min_words:
                            return False
                        return True

                    full_candidates = candidates[candidates.apply(_is_full_sentence)]
                    if not full_candidates.empty:
                        # choose the shortest full sentence so it's concise but complete
                        ham_example = full_candidates.loc[full_candidates.str.len().idxmin()]
                    else:
                        # If no full sentences found, prefer a moderately short cleaned example
                        max_short_len = 60
                        short_candidates = candidates[candidates.str.len() <= max_short_len]
                        if not short_candidates.empty:
                            ham_example = short_candidates.loc[short_candidates.str.len().idxmin()]
                        else:
                            # fallback: pick the longest cleaned example for readability
                            ham_example = candidates.loc[candidates.str.len().idxmax()]
    except Exception:
        # keep defaults on failure
        pass

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Use spam example"):
            st.session_state["predict_text"] = str(spam_example)
    with col_b:
        if st.button("Use ham example"):
            st.session_state["predict_text"] = str(ham_example)

    # text_area bound to session_state so button clicks update the value
    st.text_area("Enter an SMS message to classify", key="predict_text")

    if st.button("Predict"):
        text = st.session_state.get("predict_text", "")
        if model is None:
            st.error("No model available. Run training first.")
        else:
            pred = model.predict([text])[0]
            prob = None
            try:
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba([text])[0][1])
            except Exception:
                prob = None

            label = "spam" if int(pred) == 1 else "ham"

            # Colored display: spam -> red, ham -> green
            if label == "spam":
                st.error(f"Prediction: {label}")
            else:
                st.success(f"Prediction: {label}")

            # Show spam probability as metric + horizontal bar chart when available
            if prob is not None:
                try:
                    # numeric display
                    st.metric("Spam probability", f"{prob:.2%}")

                    # small horizontal bar chart (color depends on predicted label)
                    fig_prob, ax_prob = plt.subplots(figsize=(6, 0.5))
                    bar_color = "#d62728" if label == "spam" else "#2ca02c"
                    ax_prob.barh([0], [prob], color=bar_color)
                    ax_prob.set_xlim(0, 1)
                    ax_prob.set_yticks([])
                    ax_prob.set_xlabel("Spam probability")
                    # annotate value on bar
                    ax_prob.text(prob + 0.01 if prob < 0.95 else prob - 0.05, 0, f"{prob:.2%}", va="center")
                    plt.tight_layout()
                    st.pyplot(fig_prob)
                except Exception as e:
                    st.info(f"Unable to render probability chart: {e}")
            else:
                st.info("Spam probability not available for this model.")

    st.markdown("---")

    # 2) Dataset Overview
    st.header("Dataset Overview")
    try:
        df = load_data(download=False)
    except Exception:
        df = None

    if df is None:
        st.info("Dataset not available locally. Run training to download it or place CSV in data/.")
    else:
        st.subheader("Dataset sample")
        st.write(df.sample(min(sample_size, len(df))))

        # Class distribution (counts + bar chart)
        try:
            st.subheader("Class distribution")
            class_counts = df["label"].str.lower().value_counts()
            st.write(class_counts.to_frame(name="count"))
            st.bar_chart(class_counts)
        except Exception:
            st.info("Unable to compute class distribution.")

    st.markdown("---")

    # 3) Top Tokens by Class
    st.header("Top Tokens by Class")
    try:
        df_tokens = df if 'df' in locals() and df is not None else load_data(download=False)
    except Exception:
        df_tokens = None

    if df_tokens is None:
        st.info("Top tokens require the dataset. Run training or provide data/CSV.")
    else:
        # allow user to choose Top-N
        top_n = st.slider("Top N tokens", min_value=1, max_value=30, value=10, step=1)
        try:
            from src.preprocessing import build_vectorizer
            import numpy as _np
            import pandas as _pd
            import matplotlib.pyplot as _plt

            vec = build_vectorizer()
            X = vec.fit_transform(df_tokens["message"].astype(str).values)
            feature_names = _np.array(vec.get_feature_names_out())
            labels = df_tokens["label"].str.lower().values

            def top_tokens_for_class(class_name, top_n=10):
                mask = labels == class_name
                if mask.sum() == 0:
                    return _pd.DataFrame(columns=["token", "score"])
                Xc = X[mask]
                scores = _np.asarray(Xc.sum(axis=0)).ravel()
                top_idx = scores.argsort()[::-1][:top_n]
                return _pd.DataFrame({"token": feature_names[top_idx], "score": scores[top_idx]})

            df_spam = top_tokens_for_class("spam", top_n=top_n)
            df_ham = top_tokens_for_class("ham", top_n=top_n)

            col_spam, col_ham = st.columns(2)
            with col_spam:
                st.subheader(f"Spam — top {top_n} tokens")
                if df_spam.empty:
                    st.info("No spam examples in dataset.")
                else:
                    # horizontal bar chart, largest on top
                    fig, ax = _plt.subplots(figsize=(6, max(2, 0.3 * top_n)))
                    ax.barh(df_spam["token"][::-1], df_spam["score"][::-1], color="#d62728")
                    ax.set_xlabel("TF-IDF total score")
                    ax.set_ylabel("Token")
                    ax.set_title("Spam tokens")
                    _plt.tight_layout()
                    st.pyplot(fig)

            with col_ham:
                st.subheader(f"Ham — top {top_n} tokens")
                if df_ham.empty:
                    st.info("No ham examples in dataset.")
                else:
                    fig2, ax2 = _plt.subplots(figsize=(6, max(2, 0.3 * top_n)))
                    ax2.barh(df_ham["token"][::-1], df_ham["score"][::-1], color="#1f77b4")
                    ax2.set_xlabel("TF-IDF total score")
                    ax2.set_ylabel("Token")
                    ax2.set_title("Ham tokens")
                    _plt.tight_layout()
                    st.pyplot(fig2)

        except Exception as e:
            st.info(f"Unable to compute top tokens: {e}")

    st.markdown("---")

    # 4) Model Performance
    st.header("Model Performance")
    if model is None:
        st.info("No trained model available. Run training first to view performance plots.")
    else:
        # Try to compute on whole dataset if available
        try:
            df = df if 'df' in locals() and df is not None else load_data(download=False)
        except Exception:
            df = None

        if df is None:
            st.info("Dataset not available locally to compute live performance. Saved images (if any) are shown below.")
            show_image(CONF_MATRIX_PATH, "Confusion Matrix")
            show_image(ROC_PATH, "ROC Curve")
        else:
            messages = df["message"].astype(str).values
            y_true = (df["label"].str.lower() == "spam").astype(int).values
            try:
                probs = model.predict_proba(messages)[:, 1]
            except Exception:
                probs = None

            # Confusion / ROC / PR: show three small plots side-by-side
            if probs is not None:
                y_pred = (probs >= threshold).astype(int)
                img_c1, img_c2, img_c3 = st.columns(3)

                with img_c1:
                    fig_cm, ax_cm = plt.subplots(figsize=(3, 3))
                    plot_confusion_matrix_ax(y_true, y_pred, ax_cm)
                    st.subheader("Confusion")
                    st.pyplot(fig_cm)

                with img_c2:
                    fig_roc, ax_roc = plt.subplots(figsize=(3, 3))
                    plot_roc_ax(model, messages, y_true, ax_roc)
                    st.subheader("ROC")
                    st.pyplot(fig_roc)

                with img_c3:
                    fig_pr, ax_pr = plt.subplots(figsize=(3, 3))
                    plot_precision_recall_ax(model, messages, y_true, ax_pr)
                    st.subheader("Precision-Recall")
                    st.pyplot(fig_pr)

                # Present threshold sweep as a table (no plot)
                try:
                    import pandas as pd
                    thresholds = np.linspace(0.0, 1.0, 101)
                    precisions = []
                    recalls = []
                    f1s = []
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    for t in thresholds:
                        y_pred_t = (probs >= t).astype(int)
                        precisions.append(precision_score(y_true, y_pred_t, zero_division=0))
                        recalls.append(recall_score(y_true, y_pred_t, zero_division=0))
                        f1s.append(f1_score(y_true, y_pred_t, zero_division=0))

                    table_df = pd.DataFrame({
                        "threshold": thresholds,
                        "precision": precisions,
                        "recall": recalls,
                        "f1": f1s,
                    })

                    # Plot threshold sweep (precision / recall / f1)
                    st.subheader("Threshold sweep (precision / recall / f1)")
                    try:
                        fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
                        ax_ts.plot(table_df["threshold"], table_df["precision"], label="Precision", color="#1f77b4")
                        ax_ts.plot(table_df["threshold"], table_df["recall"], label="Recall", color="#ff7f0e")
                        ax_ts.plot(table_df["threshold"], table_df["f1"], label="F1", color="#2ca02c")
                        ax_ts.set_xlabel("Threshold")
                        ax_ts.set_ylabel("Score")
                        ax_ts.set_ylim(0.0, 1.05)
                        ax_ts.grid(alpha=0.2)
                        # mark current threshold
                        ax_ts.axvline(threshold, color="gray", linestyle="--", linewidth=1)
                        ax_ts.legend(loc="best")
                        plt.tight_layout()
                        st.pyplot(fig_ts)

                        # Show a representative set of thresholds inside a user-selected range
                        st.subheader("Threshold table (show 10 thresholds in selected range)")
                        try:
                            # allow user to pick a start/end range (defaults 0.30 -> 0.80)
                            range_start, range_end = st.slider(
                                "Select threshold range",
                                0.0,
                                1.0,
                                (0.3, 0.8),
                                step=0.01,
                            )

                            # Fixed to 10 thresholds per user's request; generate evenly spaced values
                            n_points = 10
                            selected_thresholds = np.linspace(range_start, range_end, n_points)

                            # find nearest rows in table_df (table_df uses 0.00..1.00 step=0.01)
                            arr = table_df["threshold"].to_numpy()
                            idxs = [int(np.abs(arr - float(s)).argmin()) for s in selected_thresholds]

                            # remove duplicates while preserving order (can happen when range small)
                            seen = set()
                            uniq_idxs = []
                            for i in idxs:
                                if i not in seen:
                                    seen.add(i)
                                    uniq_idxs.append(i)

                            display_df = table_df.iloc[uniq_idxs].reset_index(drop=True)
                            st.dataframe(display_df)
                            if len(display_df) > 0:
                                st.caption(
                                    f"Showing {len(display_df)} thresholds between {display_df['threshold'].iloc[0]:.2f}–{display_df['threshold'].iloc[-1]:.2f} (selected {n_points} points)")
                            else:
                                st.caption("No thresholds to display.")
                        except Exception as e:
                            st.info(f"Unable to compute selected threshold rows: {e}")
                            st.dataframe(table_df.head(10))
                    except Exception as e:
                        st.info(f"Unable to render threshold sweep plot: {e}")
                except Exception:
                    st.info("Unable to compute threshold table in this environment.")
            else:
                st.info("Model does not provide probabilities; showing saved images if available.")
                show_image(CONF_MATRIX_PATH, "Confusion Matrix")
                show_image(ROC_PATH, "ROC Curve")

    st.markdown("---")
    st.caption("Dashboard: adjust the threshold on the left to see how metrics and the confusion matrix change.")


if __name__ == "__main__":
    main()
