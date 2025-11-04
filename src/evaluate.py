import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_confusion_matrix(y_true, y_pred, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title("Confusion Matrix")
    plt.savefig(out_path)
    plt.close()


def save_roc_curve(estimator, X, y, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        RocCurveDisplay.from_estimator(estimator, X, y)
        plt.title("ROC Curve")
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print("Could not create ROC curve:", e)
