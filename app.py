import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessor import preprocess
from src.features import transform, feature_info, load_vectorizer
from src.model import load_model, predict, predict_proba

app = Flask(__name__)

# Global state for loaded model and vectorizer
CURRENT_CLF_NAME = "logistic_regression"
CURRENT_VEC_NAME = "tfidf"
PREDICTOR = None

class SentimentPredictor:
    def __init__(self, clf, vectorizer, labels, clf_name, vec_name):
        self.clf = clf
        self.vectorizer = vectorizer
        self.labels = labels
        self.clf_name = clf_name
        self.vec_name = vec_name

    def predict_one(self, raw_text: str) -> dict:
        cleaned = preprocess(raw_text)
        X = transform(self.vectorizer, [cleaned])
        label = predict(self.clf, X)[0]
        proba = predict_proba(self.clf, X)

        result = {
            "text": raw_text,
            "cleaned": cleaned,
            "label": label,
            "confidence": None,
            "probabilities": {}
        }
        if proba is not None:
            classes = self.clf.classes_
            result["probabilities"] = dict(zip(classes, proba[0].tolist()))
            result["confidence"] = float(proba[0].max())

        return result

def load_sentiment_model(clf_name="logistic_regression", vec_name="tfidf", force_train=False):
    global PREDICTOR, CURRENT_CLF_NAME, CURRENT_VEC_NAME
    
    # Map shortcuts
    aliases = {"nb": "naive_bayes", "lr": "logistic_regression"}
    clf_name = aliases.get(clf_name.lower(), clf_name)
    vec_name = vec_name.lower()
    
    clf_path = PROJECT_ROOT / "models" / f"{clf_name}.pkl"
    vec_path = PROJECT_ROOT / "models" / f"vectorizer_{vec_name}.pkl"

    if force_train or not clf_path.exists() or not vec_path.exists():
        print(f"[app] Model/vectorizer not found or retrain forced. Running training pipeline...")
        from train import run_pipeline
        result = run_pipeline(classifier_name=clf_name, vectorizer_name=vec_name, run_cv=False)
        clf = result["classifier"]
        vectorizer = result["vectorizer"]
        labels = result["labels"]
    else:
        clf = load_model(clf_path)
        vectorizer = load_vectorizer(vec_path)
        labels = list(clf.classes_)

    CURRENT_CLF_NAME = clf_name
    CURRENT_VEC_NAME = vec_name
    PREDICTOR = SentimentPredictor(clf, vectorizer, labels, clf_name, vec_name)
    return PREDICTOR

# Load model on startup
try:
    load_sentiment_model()
except Exception as e:
    print(f"Error loading model on startup: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def status():
    try:
        vocab_size = len(PREDICTOR.vectorizer.vocabulary_) if PREDICTOR else 0
        return jsonify({
            "status": "loaded" if PREDICTOR else "not_loaded",
            "classifier": CURRENT_CLF_NAME,
            "vectorizer": CURRENT_VEC_NAME,
            "vocabulary_size": vocab_size
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not PREDICTOR:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        res = PREDICTOR.predict_one(text)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict_batch", methods=["POST"])
def api_predict_batch():
    if not PREDICTOR:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json() or {}
    texts = data.get("texts", [])
    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "Invalid or empty texts array"}), 400

    try:
        results = [PREDICTOR.predict_one(t) for t in texts if t.strip()]
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            lbl = r["label"]
            if lbl in counts:
                counts[lbl] += 1
                
        return jsonify({
            "results": results,
            "summary": counts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def api_train():
    data = request.get_json() or {}
    clf_name = data.get("classifier", "logistic_regression")
    vec_name = data.get("vectorizer", "tfidf")
    
    try:
        # Run pipeline and reload
        from train import run_pipeline
        result = run_pipeline(classifier_name=clf_name, vectorizer_name=vec_name, run_cv=True)
        load_sentiment_model(clf_name=clf_name, vec_name=vec_name)
        
        metrics = result["metrics"]
        return jsonify({
            "success": True,
            "classifier": CURRENT_CLF_NAME,
            "vectorizer": CURRENT_VEC_NAME,
            "metrics": {
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
                "f1_weighted": float(metrics["f1_weighted"])
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use 5001 to avoid default MacOS AirPlay receiver port conflict on 5000
    app.run(host="127.0.0.1", port=5001, debug=True)
