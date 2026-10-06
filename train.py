import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import re

df = pd.read_csv("data/Tweets.csv")
df = df[["text", "airline_sentiment"]]
df = df[df["airline_sentiment"] != "neutral"]
df = df.sample(n=min(5000, len(df)), random_state=42)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|[^a-z\s]", "", text)
    return text

df["clean_text"] = df["text"].apply(clean_text)
df["label"] = df["airline_sentiment"].map({"negative": 0, "positive": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_vec, y_train)

nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

for name, model in [("Logistic Regression", log_reg), ("Naive Bayes", nb)]:
    preds = model.predict(X_test_vec)
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

import joblib
joblib.dump(log_reg, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel and vectorizer saved.")