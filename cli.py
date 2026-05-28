"""
cli.py
------
Interactive command-line interface for sentiment prediction.

Modes
─────
  Interactive REPL   : python cli.py
  Single prediction  : python cli.py --text "Your text here"
  Batch from file    : python cli.py --file reviews.txt
  Train first        : python cli.py --train

Colour-coded output:
  🟢 positive  →  green
  🔴 negative  →  red
  🟡 neutral   →  yellow
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ── ANSI colours (degrade gracefully on Windows cmd) ──────────────────────
class Colour:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

def _coloured(text: str, colour: str) -> str:
    try:
        return f"{colour}{text}{Colour.RESET}"
    except Exception:
        return text

LABEL_COLOUR = {
    "positive": Colour.GREEN,
    "negative": Colour.RED,
    "neutral":  Colour.YELLOW,
}


# ── Sentiment predictor ────────────────────────────────────────────────────

class SentimentPredictor:
    """Wraps a fitted classifier + vectorizer for single / batch prediction."""

    def __init__(self, clf, vectorizer, labels: list[str]):
        self.clf        = clf
        self.vectorizer = vectorizer
        self.labels     = labels

    def predict_one(self, raw_text: str) -> dict:
        """
        Predict sentiment for a single raw text string.

        Returns
        -------
        dict with keys: text, label, confidence, probabilities
        """
        from src.preprocessor import preprocess
        from src.features     import transform
        from src.model        import predict, predict_proba

        cleaned = preprocess(raw_text)
        X       = transform(self.vectorizer, [cleaned])
        label   = predict(self.clf, X)[0]
        proba   = predict_proba(self.clf, X)

        result = {"text": raw_text, "label": label, "confidence": None, "probabilities": {}}
        if proba is not None:
            classes = self.clf.classes_
            result["probabilities"] = dict(zip(classes, proba[0].tolist()))
            result["confidence"]    = float(proba[0].max())

        return result

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict_one(t) for t in texts]


def _format_result(result: dict) -> str:
    label  = result["label"]
    colour = LABEL_COLOUR.get(label, Colour.CYAN)
    conf   = result["confidence"]

    lines = [
        "",
        f"  Text       : {_coloured(result['text'][:120], Colour.BOLD)}",
        f"  Sentiment  : {_coloured(label.upper(), colour + Colour.BOLD)}",
    ]
    if conf is not None:
        bar_filled = int(conf * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        lines.append(f"  Confidence : {conf*100:5.1f}%  [{bar}]")

    if result.get("probabilities"):
        lines.append("  Breakdown  :")
        for lbl, prob in sorted(result["probabilities"].items()):
            c   = LABEL_COLOUR.get(lbl, Colour.CYAN)
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            lines.append(
                f"    {lbl:<12} {prob*100:5.1f}%  "
                f"[{_coloured(bar, c)}]"
            )

    lines.append("")
    return "\n".join(lines)


# ── Model loading / training helpers ──────────────────────────────────────

def _load_or_train(
    clf_name: str = "logistic_regression",
    vec_name: str = "tfidf",
    force_train: bool = False,
) -> SentimentPredictor:
    from src.model    import load_model
    from src.features import load_vectorizer

    clf_path = Path("models") / f"{clf_name}.pkl"
    vec_path = Path("models") / f"vectorizer_{vec_name}.pkl"

    if force_train or not clf_path.exists() or not vec_path.exists():
        print(_coloured("\n[cli] Model not found — running training pipeline...\n", Colour.YELLOW))
        from train import run_pipeline
        result = run_pipeline(classifier_name=clf_name, vectorizer_name=vec_name)
        clf        = result["classifier"]
        vectorizer = result["vectorizer"]
        labels     = result["labels"]
    else:
        clf        = load_model(clf_path)
        vectorizer = load_vectorizer(vec_path)
        labels     = list(clf.classes_)

    return SentimentPredictor(clf, vectorizer, labels)


# ── REPL ───────────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════╗
║        SENTIMENT ANALYSIS TOOL  v1.0                ║
║  Type text and press Enter to predict sentiment.    ║
║  Commands:  :quit  :train  :help  :batch <file>     ║
╚══════════════════════════════════════════════════════╝
"""

def repl(predictor: SentimentPredictor) -> None:
    print(_coloured(BANNER, Colour.CYAN))
    while True:
        try:
            user_input = input(_coloured(">> ", Colour.BOLD + Colour.CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[cli] Bye!")
            break

        if not user_input:
            continue

        # ── Built-in commands ──────────────────────────────────────────────
        if user_input.lower() in (":quit", ":q", "exit", "quit"):
            print("[cli] Goodbye!")
            break

        elif user_input.lower() == ":help":
            print("""
  :quit           Exit the CLI
  :train          Retrain the model
  :batch <file>   Predict sentiments for lines in <file>
  Any other text  Predict sentiment
""")

        elif user_input.lower() == ":train":
            from train import run_pipeline
            run_pipeline()
            print("[cli] Model retrained. Restart to use the new model.")

        elif user_input.lower().startswith(":batch "):
            filepath = user_input[7:].strip()
            _batch_from_file(filepath, predictor)

        else:
            result = predictor.predict_one(user_input)
            print(_format_result(result))


def _batch_from_file(filepath: str, predictor: SentimentPredictor) -> None:
    path = Path(filepath)
    if not path.exists():
        print(f"[cli] File not found: {filepath}")
        return
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n[cli] Predicting {len(lines)} items from '{path.name}'\n")
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for line in lines:
        result = predictor.predict_one(line)
        label  = result["label"]
        counts[label] = counts.get(label, 0) + 1
        colour = LABEL_COLOUR.get(label, Colour.CYAN)
        label_str = f"{label.upper():<10}"
        print(f"  {_coloured(label_str, colour)}  {line[:80]}")
    print(f"\n  Summary: {counts}")


# ── CLI entrypoint ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Sentiment Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py                            # interactive REPL
  python cli.py --text "Great product!"   # single prediction
  python cli.py --file reviews.txt        # batch prediction
  python cli.py --train                   # force retrain
  python cli.py --classifier nb           # use Naive Bayes
""",
    )
    p.add_argument("--text",       help="Single text to analyse")
    p.add_argument("--file",       help="File of texts (one per line)")
    p.add_argument("--train",      action="store_true", help="Force retrain model")
    p.add_argument("--classifier", default="logistic_regression",
                   help="logistic_regression (default) | naive_bayes | nb | lr")
    p.add_argument("--vectorizer", default="tfidf", help="tfidf (default) | count")
    return p.parse_args()


def main():
    args      = parse_args()
    predictor = _load_or_train(
        clf_name    = args.classifier,
        vec_name    = args.vectorizer,
        force_train = args.train,
    )

    if args.text:
        # ── Single prediction mode ─────────────────────────────────────────
        result = predictor.predict_one(args.text)
        print(_format_result(result))

    elif args.file:
        # ── Batch file mode ────────────────────────────────────────────────
        _batch_from_file(args.file, predictor)

    else:
        # ── Interactive REPL ───────────────────────────────────────────────
        repl(predictor)


if __name__ == "__main__":
    main()
