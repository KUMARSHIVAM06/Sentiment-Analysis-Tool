"""
data_loader.py
--------------
Load labeled sentiment data from CSV, apply preprocessing,
and return train / test splits ready for the pipeline.
"""

import csv
import random
from pathlib import Path

from src.preprocessor import preprocess


def load_csv(path: str | Path, text_col: str = "text", label_col: str = "label"):
    """
    Read a CSV with *text_col* and *label_col* columns.

    Returns
    -------
    texts  : list of raw strings
    labels : list of label strings
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_col, "").strip()
            label = row.get(label_col, "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)

    print(f"[data] Loaded {len(texts)} samples from '{path.name}'")
    _label_distribution(labels)
    return texts, labels


def preprocess_texts(texts: list[str], remove_stopwords: bool = True) -> list[str]:
    """Apply the full preprocessing pipeline to every sample."""
    cleaned = [preprocess(t, remove_stopwords=remove_stopwords) for t in texts]
    print(f"[data] Preprocessed {len(cleaned)} texts")
    return cleaned


def train_test_split(
    texts: list[str],
    labels: list[str],
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Stratified train / test split (no scikit-learn dependency for the split
    itself – we do it manually to keep this module lightweight).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    rng = random.Random(random_state)

    # Group indices by label for stratification
    label_indices: dict[str, list[int]] = {}
    for i, lbl in enumerate(labels):
        label_indices.setdefault(lbl, []).append(i)

    train_idx, test_idx = [], []
    for lbl, idxs in label_indices.items():
        idxs_shuffled = idxs[:]
        rng.shuffle(idxs_shuffled)
        n_test = max(1, round(len(idxs_shuffled) * test_size))
        test_idx.extend(idxs_shuffled[:n_test])
        train_idx.extend(idxs_shuffled[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = [texts[i] for i in train_idx]
    X_test  = [texts[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test  = [labels[i] for i in test_idx]

    print(
        f"[data] Split → train={len(X_train)}, test={len(X_test)} "
        f"(test_size={test_size:.0%})"
    )
    return X_train, X_test, y_train, y_test


def _label_distribution(labels: list[str]) -> None:
    from collections import Counter
    dist = Counter(labels)
    total = len(labels)
    print("[data] Label distribution:")
    for lbl, cnt in sorted(dist.items()):
        bar = "█" * int(cnt / total * 30)
        print(f"       {lbl:<12} {cnt:>4}  ({cnt/total*100:5.1f}%)  {bar}")
