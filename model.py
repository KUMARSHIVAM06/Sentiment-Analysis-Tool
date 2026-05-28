"""
model.py
--------
Train, evaluate, and persist sentiment classifiers.

Supported classifiers
─────────────────────
  • Naive Bayes        (MultinomialNB  – fast, strong baseline)
  • Logistic Regression (powerful, handles TF-IDF well)
"""

import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


# ── Classifier registry ────────────────────────────────────────────────────

CLASSIFIERS = {
    "naive_bayes": MultinomialNB(alpha=1.0),
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    ),
}


def get_classifier(name: str):
    """Return a fresh (unfitted) classifier by name."""
    name = name.lower().replace(" ", "_")
    if name not in CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier '{name}'. "
            f"Choose from: {list(CLASSIFIERS.keys())}"
        )
    # Return a clone so the registry stays pristine
    from sklearn.base import clone
    return clone(CLASSIFIERS[name])


# ── Training ───────────────────────────────────────────────────────────────

def train(clf, X_train, y_train):
    """Fit *clf* on training data and return it."""
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f"[model] Training complete in {elapsed:.3f}s")
    return clf


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(clf, X_test, y_test, labels: list[str] | None = None) -> dict:
    """
    Compute and print evaluation metrics.

    Returns
    -------
    dict with keys: accuracy, f1_macro, f1_weighted, report, confusion_matrix
    """
    y_pred = clf.predict(X_test)

    acc        = accuracy_score(y_test, y_pred)
    f1_macro   = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report     = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
    cm         = confusion_matrix(y_test, y_pred)

    print("\n" + "─" * 50)
    print("  EVALUATION RESULTS")
    print("─" * 50)
    print(f"  Accuracy       : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 (macro)     : {f1_macro:.4f}")
    print(f"  F1 (weighted)  : {f1_weighted:.4f}")
    print("\n  Classification Report:")
    print(report)

    if labels:
        _print_confusion_matrix(cm, labels)

    return {
        "accuracy":     acc,
        "f1_macro":     f1_macro,
        "f1_weighted":  f1_weighted,
        "report":       report,
        "confusion_matrix": cm,
    }


def cross_validate(clf, X, y, n_splits: int = 5) -> dict:
    """
    Run stratified k-fold cross-validation and print summary.

    Returns mean ± std for accuracy and F1-macro.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    f1_scores  = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")

    print(f"\n[model] {n_splits}-Fold Cross-Validation")
    print(f"  Accuracy : {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"  F1-macro : {f1_scores.mean():.4f}  ± {f1_scores.std():.4f}")

    return {
        "cv_accuracy_mean": acc_scores.mean(),
        "cv_accuracy_std":  acc_scores.std(),
        "cv_f1_mean":       f1_scores.mean(),
        "cv_f1_std":        f1_scores.std(),
    }


# ── Prediction ─────────────────────────────────────────────────────────────

def predict(clf, X) -> np.ndarray:
    """Return predicted class labels."""
    return clf.predict(X)


def predict_proba(clf, X) -> np.ndarray | None:
    """Return class probabilities if the classifier supports it."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    return None


# ── Persistence ────────────────────────────────────────────────────────────

def save_model(clf, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[model] Model saved → {path}")


def load_model(path: str | Path):
    with open(path, "rb") as f:
        clf = pickle.load(f)
    print(f"[model] Model loaded ← {path}")
    return clf


# ── Helpers ────────────────────────────────────────────────────────────────

def _print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    col_w = max(len(l) for l in labels) + 2
    header = "Predicted →".ljust(col_w) + "  ".join(l.center(col_w) for l in labels)
    print("  Confusion Matrix:")
    print("  " + header)
    for i, row_label in enumerate(labels):
        row = row_label.ljust(col_w) + "  ".join(str(v).center(col_w) for v in cm[i])
        print("  " + row)
    print()
