# Sentiment Analysis on Product Reviews

Classifies text as positive or negative using TF-IDF and classic ML models.

## Dataset
Twitter US Airline Sentiment dataset (Kaggle), sampled to ~5000 rows.

## Approach
1. Cleaned text (lowercase, removed URLs/mentions/punctuation)
2. TF-IDF features (3000 features, English stopwords removed)
3. Trained and compared Logistic Regression and Naive Bayes
4. Evaluated with accuracy, precision, recall, confusion matrix

## Results
- Logistic Regression: accuracy ___%, precision ___%, recall ___%
- Naive Bayes: accuracy 85.1%, precision 95.7%, recall 31.5%

## How to run
pip install -r requirements.txt
python train.py
python predict.py