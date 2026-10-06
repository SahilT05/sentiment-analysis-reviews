import joblib
import re

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|[^a-z\s]", "", text)
    return text

while True:
    text = input("\nEnter a review (or 'quit'): ")
    if text.lower() == "quit":
        break
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    print("Prediction:", "Positive" if pred == 1 else "Negative")