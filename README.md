# Sentiment Analysis Tool

A complete, production-style sentiment analysis project built with **scikit-learn**.

---

## Features

| Capability | Details |
|---|---|
| Data ingestion | CSV loader with automatic label distribution report |
| Text preprocessing | Lowercasing, contraction expansion, URL/HTML removal, stopword filtering |
| Feature extraction | TF-IDF (default) **or** CountVectorizer with bigrams |
| Classifiers | Naive Bayes (fast baseline) **or** Logistic Regression (best accuracy) |
| Evaluation | Accuracy, F1-macro, F1-weighted, classification report, confusion matrix |
| Cross-validation | Stratified 5-fold CV |
| Persistence | Pickle save/load for both model and vectorizer |
| CLI | Interactive REPL, single-text mode, batch file mode |

---

## Project Structure

```
sentiment_analysis/
├── data/
│   └── sentiment_data.csv          # Labeled tweets / reviews (pos / neg / neutral)
├── models/                         # Saved .pkl files (created after training)
├── src/
│   ├── __init__.py
│   ├── preprocessor.py             # Text cleaning & tokenization
│   ├── features.py                 # CountVectorizer / TF-IDF wrappers
│   ├── data_loader.py              # CSV loading & train/test split
│   └── model.py                    # Training, evaluation, persistence
├── tests/
│   └── test_sentiment.py           # Unit + integration tests
├── train.py                        # End-to-end training pipeline script
├── cli.py                          # Interactive CLI / prediction interface
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
# Default: Logistic Regression + TF-IDF
python train.py

# Use Naive Bayes
python train.py --classifier nb

# Use CountVectorizer
python train.py --vectorizer count

# Both options combined
python train.py --classifier lr --vectorizer tfidf --test-size 0.2
```

### 3. Run the CLI

```bash
# Interactive REPL (model auto-trains if not found)
python cli.py

# Single text prediction
python cli.py --text "This product is absolutely fantastic!"

# Batch prediction from a file (one text per line)
python cli.py --file reviews.txt

# Force retrain then start REPL
python cli.py --train
```

### 4. Run tests

```bash
python -m pytest tests/ -v
# or
python -m unittest tests/test_sentiment.py -v
```

---

## CLI Commands (REPL Mode)

| Command | Action |
|---|---|
| Any text | Predict sentiment |
| `:quit` or `:q` | Exit |
| `:train` | Retrain the model |
| `:batch <file>` | Batch predict from file |
| `:help` | Show help |

---

## Preprocessing Pipeline

```
Raw text
  ↓ lowercase
  ↓ expand contractions (won't → will not)
  ↓ remove URLs
  ↓ remove HTML tags
  ↓ remove @mentions and #hashtags
  ↓ remove punctuation
  ↓ remove stopwords
  ↓ collapse whitespace
Cleaned text
```

---

## Model Performance (example on included dataset)

| Classifier | Vectorizer | Accuracy | F1-macro |
|---|---|---|---|
| Logistic Regression | TF-IDF | ~0.90+ | ~0.90+ |
| Naive Bayes | TF-IDF | ~0.85+ | ~0.85+ |

*Exact results vary with random seed and dataset size.*

---

## Extending the Project

- **Add more data**: drop any CSV with `text` and `label` columns into `data/`  
- **New classifier**: register it in `src/model.py → CLASSIFIERS`  
- **Hyperparameter tuning**: use `sklearn.model_selection.GridSearchCV` around `run_pipeline()`  
- **Deep learning**: replace the sklearn vectorizer+classifier with a Hugging Face transformer
