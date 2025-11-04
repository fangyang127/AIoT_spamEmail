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


def main():
    st.set_page_config(page_title="Spam SMS Classifier", layout="wide")
    st.title("Spam SMS Classifier — Demo")

    import streamlit as st
    from joblib import load
    import os
    import sys
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        ConfusionMatrixDisplay,
        RocCurveDisplay,
    )

    # Ensure repository root is on sys.path so `from src...` imports work on Streamlit Cloud
    # app/streamlit_app.py lives in <repo>/app/ ; repo root is one level up
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from src.data_loader import load_data

    MODEL_PATH = os.path.join("models", "model--logistic--v0.1.joblib")
    METRICS_PATH = os.path.join("models", "metrics.json")
    METADATA_PATH = os.path.join("models", "metadata.json")


    def load_model(path=MODEL_PATH):
        if not os.path.exists(path):
            return None
        try:
            return load(path)
        except Exception:
            return None


    def load_json(path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


    def plot_confusion_matrix_ax(y_true, y_pred, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
        return ax


    def plot_roc_ax(estimator, X, y, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        try:
            RocCurveDisplay.from_estimator(estimator, X, y, ax=ax)
        except Exception as e:
            ax.text(0.5, 0.5, f"ROC not available: {e}", ha="center")
        return ax


    def main():
        st.set_page_config(page_title="Spam SMS Classifier", layout="wide")

        # Sidebar controls
        st.sidebar.title("Controls")
        threshold = st.sidebar.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.01)
        sample_size = st.sidebar.slider("Dataset sample size (for preview)", 1, 50, 10)
        reload_model = st.sidebar.button("Reload model")
        show_prob_hist = st.sidebar.checkbox("Show probability histogram", True)
        show_raw_metrics = st.sidebar.checkbox("Show saved metrics.json", True)

        st.title("Spam SMS Classifier — Dashboard")

        # Load model and data
        if reload_model:
            try:
                st.experimental_memo_clear()
            except Exception:
                pass

        model = load_model()
        metrics = load_json(METRICS_PATH)
        metadata = load_json(METADATA_PATH)

        col1, col2 = st.columns([2, 3])

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

            st.markdown("---")
            st.header("Model info")
            if metadata:
                st.subheader("Model metadata")
                st.json(metadata)
            else:
                st.info("No metadata found. Consider generating metadata in training.")

            if show_raw_metrics and metrics:
                st.subheader("Saved metrics")
                st.json(metrics)

        with col2:
            st.header("Dataset & Evaluation")
            # Try to load dataset
            try:
                df = load_data(download=False)
            except Exception:
                df = None

            if df is None:
                st.info("Dataset not available locally. Run training to download it or place CSV in data/.")
            else:
                st.subheader("Dataset sample")
                st.write(df.sample(min(sample_size, len(df))))

                if model is None:
                    st.info("No trained model available to compute live evaluation.")
                else:
                    # Compute probabilities and metrics on whole dataset (could be expensive)
                    messages = df["message"].astype(str).values
                    y_true = (df["label"].str.lower() == "spam").astype(int).values
                    try:
                        probs = model.predict_proba(messages)[:, 1]
                    except Exception:
                        probs = None

                    if probs is not None:
                        y_pred = (probs >= threshold).astype(int)
                        acc = accuracy_score(y_true, y_pred)
                        prec = precision_score(y_true, y_pred, zero_division=0)
                        rec = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        st.subheader("Live metrics (based on full dataset)")
                        st.metric("Accuracy", f"{acc:.3f}")
                        st.metric("Precision", f"{prec:.3f}")
                        st.metric("Recall", f"{rec:.3f}")
                        st.metric("F1", f"{f1:.3f}")

                        # Probability histogram
                        if show_prob_hist:
                            fig, ax = plt.subplots()
                            ax.hist(probs, bins=30, color="#2b8cbe", alpha=0.8)
                            ax.set_xlabel("Spam probability")
                            ax.set_ylabel("Count")
                            st.pyplot(fig)

                        # Confusion matrix and ROC
                        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                        plot_confusion_matrix_ax(y_true, y_pred, ax_cm)
                        st.pyplot(fig_cm)

                        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                        plot_roc_ax(model, messages, y_true, ax_roc)
                        st.pyplot(fig_roc)
                    else:
                        st.info("Model does not provide probabilities; dynamic thresholding is not available.")

        st.markdown("---")
        st.caption("Dashboard: adjust the threshold on the left to see how metrics and the confusion matrix change.")


    if __name__ == "__main__":
        main()
