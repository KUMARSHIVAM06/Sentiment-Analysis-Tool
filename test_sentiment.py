"""
tests/test_sentiment.py
-----------------------
Unit tests covering preprocessor, feature extraction, model,
data loader, and the end-to-end pipeline.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import (
    expand_contractions,
    preprocess,
    remove_html_tags,
    remove_mentions_hashtags,
    remove_urls,
    tokenize,
)
from src.features import build_vectorizer, feature_info, fit_transform, transform
from src.data_loader import preprocess_texts, train_test_split
from src.model import get_classifier, train, evaluate, predict


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessor(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(preprocess("HELLO WORLD"), "hello world")

    def test_expand_contractions(self):
        result = expand_contractions("I won't and can't")
        self.assertIn("will not", result)
        self.assertIn("cannot", result)

    def test_remove_urls(self):
        text = "Visit https://example.com for more"
        self.assertNotIn("http", remove_urls(text))

    def test_remove_html(self):
        text = "<b>Bold</b> and <i>italic</i>"
        cleaned = remove_html_tags(text)
        self.assertNotIn("<", cleaned)
        self.assertIn("Bold", cleaned)

    def test_remove_mentions_hashtags(self):
        text = "Hey @user check #trending"
        cleaned = remove_mentions_hashtags(text)
        self.assertNotIn("@user", cleaned)
        self.assertNotIn("#trending", cleaned)

    def test_tokenize_removes_stopwords(self):
        tokens = tokenize("this is a great product")
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("a", tokens)
        self.assertIn("great", tokens)
        self.assertIn("product", tokens)

    def test_preprocess_returns_string(self):
        result = preprocess("Great product! Highly recommend.")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_preprocess_empty_string(self):
        result = preprocess("")
        self.assertEqual(result, "")

    def test_preprocess_only_stopwords(self):
        result = preprocess("this is a the")
        self.assertEqual(result.strip(), "")


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction tests
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_TEXTS = [
    "love product amazing",
    "terrible experience horrible",
    "okay average nothing special",
    "excellent quality highly recommend",
    "worst product ever avoid",
]

class TestFeatures(unittest.TestCase):

    def test_tfidf_shape(self):
        _, X = fit_transform(_SAMPLE_TEXTS, method="tfidf")
        self.assertEqual(X.shape[0], len(_SAMPLE_TEXTS))
        self.assertGreater(X.shape[1], 0)

    def test_count_shape(self):
        _, X = fit_transform(_SAMPLE_TEXTS, method="count")
        self.assertEqual(X.shape[0], len(_SAMPLE_TEXTS))

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            build_vectorizer(method="unknown")

    def test_transform_new_texts(self):
        vec, X_train = fit_transform(_SAMPLE_TEXTS, method="tfidf")
        X_new = transform(vec, ["brand new text here"])
        self.assertEqual(X_new.shape[0], 1)
        self.assertEqual(X_new.shape[1], X_train.shape[1])

    def test_feature_info_keys(self):
        vec, _ = fit_transform(_SAMPLE_TEXTS, method="tfidf")
        info = feature_info(vec)
        for key in ("type", "vocabulary_size", "ngram_range", "top_20_terms"):
            self.assertIn(key, info)


# ─────────────────────────────────────────────────────────────────────────────
# Data loader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoader(unittest.TestCase):

    def _sample(self):
        texts  = ["Great!", "Terrible.", "Okay.", "Amazing!", "Awful."]
        labels = ["positive", "negative", "neutral", "positive", "negative"]
        return texts, labels

    def test_preprocess_texts_length(self):
        texts, _ = self._sample()
        cleaned  = preprocess_texts(texts)
        self.assertEqual(len(cleaned), len(texts))

    def test_train_test_split_sizes(self):
        texts, labels = self._sample()
        X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2)
        self.assertEqual(len(X_tr) + len(X_te), len(texts))
        self.assertGreaterEqual(len(X_te), 1)

    def test_train_test_split_no_data_leakage(self):
        texts  = [str(i) for i in range(20)]
        labels = ["positive"] * 10 + ["negative"] * 10
        X_tr, X_te, _, _ = train_test_split(texts, labels, test_size=0.2)
        self.assertEqual(len(set(X_tr) & set(X_te)), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_TEXTS  = [
    "love amazing wonderful great excellent",
    "terrible awful horrible worst never",
    "okay average fine decent neutral",
    "fantastic superb brilliant outstanding",
    "dreadful dreadful useless waste money",
    "mediocre passable acceptable nothing special",
    "best purchase happy satisfied recommend",
    "disappointed regret broken defective",
    "ordinary regular standard basic normal",
    "perfect wonderful love recommend buy",
    "poor quality cheap flimsy bad",
    "alright reasonable moderate mid range",
]
_TRAIN_LABELS = [
    "positive", "negative", "neutral",
    "positive", "negative", "neutral",
    "positive", "negative", "neutral",
    "positive", "negative", "neutral",
]

class TestModel(unittest.TestCase):

    def setUp(self):
        self.vec, self.X = fit_transform(_TRAIN_TEXTS, method="tfidf")
        self.labels      = _TRAIN_LABELS

    def test_naive_bayes_trains(self):
        clf = get_classifier("naive_bayes")
        clf = train(clf, self.X, self.labels)
        self.assertTrue(hasattr(clf, "classes_"))

    def test_logistic_regression_trains(self):
        clf = get_classifier("logistic_regression")
        clf = train(clf, self.X, self.labels)
        self.assertTrue(hasattr(clf, "classes_"))

    def test_predict_returns_labels(self):
        clf = get_classifier("logistic_regression")
        clf = train(clf, self.X, self.labels)
        preds = predict(clf, self.X)
        self.assertEqual(len(preds), len(self.labels))

    def test_evaluate_returns_metrics(self):
        clf = get_classifier("logistic_regression")
        clf = train(clf, self.X, self.labels)
        metrics = evaluate(clf, self.X, self.labels)
        self.assertIn("accuracy",   metrics)
        self.assertIn("f1_macro",   metrics)
        self.assertIn("f1_weighted", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(   metrics["accuracy"], 1.0)

    def test_unknown_classifier_raises(self):
        with self.assertRaises(ValueError):
            get_classifier("unknown_clf")


# ─────────────────────────────────────────────────────────────────────────────
# Integration: end-to-end pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd(unittest.TestCase):

    def test_pipeline_runs(self):
        from train import run_pipeline
        result = run_pipeline(
            data_path       = "data/sentiment_data.csv",
            classifier_name = "logistic_regression",
            vectorizer_name = "tfidf",
            test_size       = 0.25,
            run_cv          = False,
        )
        self.assertIn("metrics", result)
        self.assertGreater(result["metrics"]["accuracy"], 0.4)

    def test_predictor_returns_valid_label(self):
        from train import run_pipeline
        from cli   import SentimentPredictor

        result    = run_pipeline(data_path="data/sentiment_data.csv",
                                 run_cv=False)
        predictor = SentimentPredictor(
            result["classifier"], result["vectorizer"], result["labels"]
        )
        out = predictor.predict_one("This product is absolutely wonderful!")
        self.assertIn(out["label"], ["positive", "negative", "neutral"])
        self.assertIn("confidence", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
