"""
train.py
--------
End-to-end training pipeline:
  1. Load & preprocess data
  2. Extract features (TF-IDF or CountVectorizer)
  3. Train classifier (Naive Bayes or Logistic Regression)
  4. Evaluate on held-out test set
  5. Save model + vectorizer artifacts

Usage
─────
  python train.py                          # defaults
  python train.py --classifier nb          # naive_bayes
  python train.py --classifier lr          # logistic_regression
  python train.py --vectorizer count       # CountVectorizer
  python train.py --data path/to/data.csv
  python train.py --no-cv                  # skip cross-validation
"""

import argparse
import sys
from pathlib import Path

# Make sure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_csv, preprocess_texts, train_test_split
from src.features import feature_info, fit_transform, save_vectorizer, transform
from src.model import (
    cross_validate,
    evaluate,
    get_classifier,
    save_model,
    train,
)

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_DATA       = "data/sentiment_data.csv"
DEFAULT_CLF        = "logistic_regression"   # "naive_bayes" | "logistic_regression"
DEFAULT_VECTORIZER = "tfidf"                 # "tfidf" | "count"
MODEL_DIR          = Path("models")


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train a sentiment classifier")
    p.add_argument("--data",       default=DEFAULT_DATA,       help="Path to CSV dataset")
    p.add_argument("--classifier", default=DEFAULT_CLF,        help="naive_bayes | logistic_regression (or nb | lr)")
    p.add_argument("--vectorizer", default=DEFAULT_VECTORIZER, help="tfidf | count")
    p.add_argument("--test-size",  type=float, default=0.20,   help="Fraction for test split (default: 0.20)")
    p.add_argument("--no-cv",      action="store_true",        help="Skip cross-validation")
    return p.parse_args()


# ── Classifier shorthand aliases ───────────────────────────────────────────
CLF_ALIASES = {"nb": "naive_bayes", "lr": "logistic_regression"}


# ── Main pipeline ──────────────────────────────────────────────────────────
def run_pipeline(
    data_path: str       = DEFAULT_DATA,
    classifier_name: str = DEFAULT_CLF,
    vectorizer_name: str = DEFAULT_VECTORIZER,
    test_size: float     = 0.20,
    run_cv: bool         = True,
) -> dict:
    """
    Execute the full training pipeline.

    Returns a dict with trained clf, vectorizer, labels, and metrics.
    """
    classifier_name = CLF_ALIASES.get(classifier_name.lower(), classifier_name)

    print("\n" + "═" * 55)
    print("  SENTIMENT ANALYSIS — TRAINING PIPELINE")
    print("═" * 55)
    print(f"  Dataset    : {data_path}")
    print(f"  Classifier : {classifier_name}")
    print(f"  Vectorizer : {vectorizer_name}")
    print(f"  Test size  : {test_size:.0%}")
    print("═" * 55 + "\n")

    # ── Step 1 : Load data ─────────────────────────────────────────────────
    raw_texts, labels = load_csv(data_path)

    # ── Step 2 : Preprocess ────────────────────────────────────────────────
    texts = preprocess_texts(raw_texts)

    # ── Step 3 : Split ────────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size
    )

    # ── Step 4 : Feature extraction ───────────────────────────────────────
    vectorizer, X_train = fit_transform(X_train_raw, method=vectorizer_name)
    X_test              = transform(vectorizer, X_test_raw)

    info = feature_info(vectorizer)
    print(f"\n[features] Vocabulary size : {info['vocabulary_size']}")
    print(f"[features] Feature matrix  : {X_train.shape}")

    # ── Step 5 : Cross-validation (optional, on full cleaned data) ────────
    all_labels = sorted(set(labels))
    if run_cv:
        vec_full, X_full = fit_transform(texts, method=vectorizer_name)
        clf_cv = get_classifier(classifier_name)
        cross_validate(clf_cv, X_full, labels, n_splits=5)

    # ── Step 6 : Train ────────────────────────────────────────────────────
    clf = get_classifier(classifier_name)
    clf = train(clf, X_train, y_train)

    # ── Step 7 : Evaluate ─────────────────────────────────────────────────
    metrics = evaluate(clf, X_test, y_test, labels=all_labels)

    # ── Step 8 : Save artifacts ───────────────────────────────────────────
    clf_path = MODEL_DIR / f"{classifier_name}.pkl"
    vec_path = MODEL_DIR / f"vectorizer_{vectorizer_name}.pkl"
    save_model(clf,        clf_path)
    save_vectorizer(vectorizer, vec_path)

    print("\n[pipeline] Done! Artifacts saved to", MODEL_DIR)
    print("═" * 55 + "\n")

    return {
        "classifier":  clf,
        "vectorizer":  vectorizer,
        "labels":      all_labels,
        "metrics":     metrics,
        "model_path":  clf_path,
        "vec_path":    vec_path,
    }


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_path       = args.data,
        classifier_name = args.classifier,
        vectorizer_name = args.vectorizer,
        test_size       = args.test_size,
        run_cv          = not args.no_cv,
    )
