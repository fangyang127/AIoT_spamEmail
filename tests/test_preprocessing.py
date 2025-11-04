from src.preprocessing import build_vectorizer


def test_vectorizer_transforms():
    v = build_vectorizer()
    sample = ["hello world", "buy cheap meds now"]
    X = v.fit_transform(sample)
    assert X.shape[0] == 2
    assert X.shape[1] >= 1
