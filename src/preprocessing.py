from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer():
    """Return a TF-IDF vectorizer configured for short-text messages."""
    return TfidfVectorizer(lowercase=True, stop_words="english", max_df=0.95, min_df=1)


def simple_tokenizer(text: str):
    # placeholder if custom tokenization needed
    return text.split()


if __name__ == "__main__":
    v = build_vectorizer()
    sample = ["hello world", "buy cheap meds now"]
    X = v.fit_transform(sample)
    print("TF-IDF shape:", X.shape)
