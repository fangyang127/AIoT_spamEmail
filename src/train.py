import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .data_loader import load_data
from .model import build_pipeline, save_model
from .evaluate import save_confusion_matrix, save_roc_curve


def train_smoke(output_dir: str = "models", test_size: float = 0.2):
    df = load_data()
    # map labels to binary
    y = (df["label"].str.lower() == "spam").astype(int)
    X = df["message"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    print("Training pipeline...")
    pipe.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            metrics["roc_auc"] = None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(output_dir, "model--logistic--v0.1.joblib")
    save_model(pipe, model_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save metadata describing the model and training run
    try:
        import sklearn
        from datetime import datetime

        metadata = {
            "model_name": "logistic",
            "model_version": "v0.1",
            "model_path": os.path.basename(model_path),
            "pipeline_steps": [name for name, _ in pipe.steps],
            "train_date": datetime.utcnow().isoformat() + "Z",
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "sklearn_version": sklearn.__version__,
            "metrics": metrics,
        }
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    except Exception as e:
        print(f"Failed to write metadata: {e}")

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    # Save evaluation plots
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    save_confusion_matrix(y_test, y_pred, cm_path)
    roc_path = os.path.join(output_dir, "roc_curve.png")
    # save_roc_curve will handle exceptions (e.g., if probs not available)
    save_roc_curve(pipe, X_test, y_test, roc_path)

    print("Metrics:", metrics)
    return model_path, metrics_path


if __name__ == "__main__":
    train_smoke()
