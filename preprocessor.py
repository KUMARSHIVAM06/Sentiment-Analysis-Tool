"""
preprocessor.py
---------------
Text cleaning and tokenization utilities for the Sentiment Analysis pipeline.
"""

import re
import string


# Common English stopwords (no external dependency)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
    "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
}

# Contractions map
CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'ve": " have", "'m": " am",
}


def expand_contractions(text: str) -> str:
    """Expand common English contractions."""
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def remove_urls(text: str) -> str:
    """Strip http/https URLs and www addresses."""
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_html_tags(text: str) -> str:
    """Remove any HTML tags."""
    return re.sub(r"<[^>]+>", "", text)


def remove_mentions_hashtags(text: str) -> str:
    """Remove Twitter-style @mentions and #hashtags."""
    return re.sub(r"[@#]\w+", "", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces and strip leading/trailing whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, dropping stopwords."""
    tokens = text.lower().split()
    return [t for t in tokens if t and t not in STOPWORDS]


def preprocess(text: str, remove_stopwords: bool = True) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Expand contractions
      3. Remove URLs, HTML, mentions/hashtags
      4. Remove punctuation
      5. Remove extra whitespace
      6. Optionally remove stopwords

    Returns a cleaned string (suitable for CountVectorizer / TF-IDF).
    """
    text = text.lower()
    text = expand_contractions(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_mentions_hashtags(text)
    text = remove_punctuation(text)
    text = remove_extra_whitespace(text)

    if remove_stopwords:
        tokens = [t for t in text.split() if t not in STOPWORDS]
        text = " ".join(tokens)

    return text


if __name__ == "__main__":
    samples = [
        "I LOVE this product! It's absolutely amazing 😍 #bestpurchase",
        "Won't buy again. Terrible experience... http://review.com",
        "<b>Great</b> quality and @seller was very helpful!",
    ]
    print("=== Preprocessor Demo ===\n")
    for s in samples:
        print(f"  Original : {s}")
        print(f"  Cleaned  : {preprocess(s)}")
        print()
