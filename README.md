
# Sentiment Analysis Tool

A Machine Learning based Sentiment Analysis Tool that classifies text reviews/tweets into Positive or Negative sentiment.

This project demonstrates:
- Text preprocessing
- Feature extraction using TF-IDF / CountVectorizer
- Model training using Naive Bayes or Logistic Regression
- Model evaluation using Accuracy and F1-score
- Command Line Interface (CLI) for real-time sentiment prediction

---

## Features

✔ Load labeled dataset (tweets/reviews)  
✔ Clean and preprocess text  
✔ Convert text into numerical vectors  
✔ Train ML classification models  
✔ Evaluate model performance  
✔ Predict sentiment from user input using CLI  

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK

---

## Project Structure

```bash
Sentiment-Analysis-Tool/
│
├── data/
│   └── reviews.csv
│
├── model/
│   └── sentiment_model.pkl
│
├── sentiment_analysis.py
├── requirements.txt
├── README.md
└── .gitignore
