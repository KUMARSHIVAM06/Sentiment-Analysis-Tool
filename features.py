"""
features.py
-----------
Convert preprocessed text into numeric feature matrices using
CountVectorizer or TF-IDF.  Wraps scikit-learn transformers with
a consistent interface and adds persistence (save / load).
"""

import pickle
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ── Default vectorizer hyper-parameters ────────────────────────────────────
_DEFAULTS = dict(
    ngram_range=(1, 2),   # unigrams + bigrams
    max_features=10_000,  # vocabulary cap
    min_df=1,             # minimum document frequency
    max_df=0.95,          # ignore terms in > 95 % of docs
    sublinear_tf=True,    # log-scale TF (TF-IDF only)
)


def build_vectorizer(method: str = "tfidf", **kwargs):
    """
    Create a fresh (unfitted) vectorizer.

    Parameters
    ----------
    method : str
        "tfidf"  → TfidfVectorizer  (default, recommended)
        "count"  → CountVectorizer
    **kwargs
        Override any scikit-learn vectorizer parameter.

    Returns
    -------
    sklearn vectorizer instance (unfitted)
    """
    params = {k: v for k, v in _DEFAULTS.items()}
    # sublinear_tf is TF-IDF-only; drop it for CountVectorizer
    if method == "count":
        params.pop("sublinear_tf", None)
    params.update(kwargs)

    method = method.lower()
    if method == "tfidf":
        return TfidfVectorizer(**params)
    elif method == "count":
        return CountVectorizer(**params)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'tfidf' or 'count'.")


def fit_transform(texts: list[str], method: str = "tfidf", **kwargs):
    """
    Fit a new vectorizer on *texts* and return (vectorizer, X).

    Parameters
    ----------
    texts  : list of preprocessed strings
    method : "tfidf" | "count"

    Returns
    -------
    vectorizer : fitted vectorizer
    X          : sparse feature matrix  (n_samples × n_features)
    """
    vec = build_vectorizer(method, **kwargs)
    X = vec.fit_transform(texts)
    return vec, X


def transform(vectorizer, texts: list[str]):
    """Apply a *fitted* vectorizer to new texts."""
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer, path: str | Path) -> None:
    """Pickle the fitted vectorizer to *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[features] Vectorizer saved → {path}")


def load_vectorizer(path: str | Path):
    """Load a pickled vectorizer from *path*."""
    with open(path, "rb") as f:
        vec = pickle.load(f)
    print(f"[features] Vectorizer loaded ← {path}")
    return vec


def feature_info(vectorizer) -> dict:
    """Return a summary dict about a fitted vectorizer."""
    vocab = vectorizer.vocabulary_
    return {
        "type": type(vectorizer).__name__,
        "vocabulary_size": len(vocab),
        "ngram_range": vectorizer.ngram_range,
        "top_20_terms": sorted(vocab, key=vocab.get)[:20],
    }


if __name__ == "__main__":
    sample_texts = [
        "love product amazing works perfectly",
        "terrible experience never buying again",
        "okay nothing special gets job done",
    ]
    print("=== Feature Extraction Demo ===\n")
    for method in ("tfidf", "count"):
        vec, X = fit_transform(sample_texts, method=method)
        info = feature_info(vec)
        print(f"  Method         : {info['type']}")
        print(f"  Vocabulary size: {info['vocabulary_size']}")
        print(f"  Feature matrix : {X.shape}")
        print(f"  Sample terms   : {info['top_20_terms'][:8]}")
        print()
