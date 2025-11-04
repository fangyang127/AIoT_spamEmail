from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump, load
from typing import Optional
from pathlib import Path

from .preprocessing import build_vectorizer


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", build_vectorizer()),
        ("clf", LogisticRegression(solver="liblinear", random_state=42)),
    ])


def save_model(pipeline: Pipeline, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, path)


def load_model(path: str) -> Pipeline:
    return load(path)


if __name__ == "__main__":
    pipe = build_pipeline()
    print(pipe)
