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

    st.set_page_config(page_title="Spam SMS Classifier", layout="wide")
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

    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        st.header("Predict a message")
        text = st.text_area("Enter an SMS message to classify", "Free entry: claim your prize now")
        if st.button("Predict"):
            if model is None:
                st.error("No model available. Run training first.")
            else:
                pred = model.predict([text])[0]
                prob = model.predict_proba([text])[0][1] if hasattr(model, "predict_proba") else None
                label = "spam" if int(pred) == 1 else "ham"
                st.markdown(f"**Prediction:** {label}")
                if prob is not None:
                    st.write(f"Spam probability: {prob:.3f}")
        # Model info removed per request — no metadata/metrics are shown here.

    with col2:
        st.header("Dataset")
        try:
            df = load_data(download=False)
        except Exception:
            df = None

        if df is None:
            st.info("Dataset not available locally. Run training to download it or place CSV in data/.")
        else:
            st.subheader("Dataset sample")
            st.write(df.sample(min(sample_size, len(df))))

    with col3:
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

                # Confusion matrix at current threshold
                if probs is not None:
                    y_pred = (probs >= threshold).astype(int)
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                    plot_confusion_matrix_ax(y_true, y_pred, ax_cm)
                    st.subheader("Confusion Matrix (current threshold)")
                    st.pyplot(fig_cm)

                    # ROC
                    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                    plot_roc_ax(model, messages, y_true, ax_roc)
                    st.subheader("ROC Curve")
                    st.pyplot(fig_roc)

                    # Precision-Recall curve
                    fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
                    plot_precision_recall_ax(model, messages, y_true, ax_pr)
                    st.subheader("Precision-Recall Curve")
                    st.pyplot(fig_pr)

                    # Threshold sweep
                    fig_ts, ax_ts = plt.subplots(figsize=(6, 3))
                    plot_threshold_sweep_ax(probs, y_true, ax_ts)
                    st.subheader("Threshold sweep (precision / recall / f1)")
                    st.pyplot(fig_ts)
                    # Also present threshold sweep as a table
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
                        st.subheader("Threshold table (showing ~10 rows around current threshold)")
                        try:
                            # show ~10 rows centered around the current slider threshold
                            import math

                            window = 10
                            center_idx = int(round(threshold * 100))
                            start = max(0, center_idx - window // 2)
                            end = start + window
                            # clamp end
                            if end > len(table_df):
                                end = len(table_df)
                                start = max(0, end - window)

                            display_df = table_df.iloc[start:end].reset_index(drop=True)
                            st.dataframe(display_df)
                            st.caption(f"Showing rows {start}–{end-1} (thresholds {display_df['threshold'].iloc[0]:.2f}–{display_df['threshold'].iloc[-1]:.2f})")
                        except Exception:
                            st.dataframe(table_df.head(10))
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
