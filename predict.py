import argparse
import joblib
import pandas as pd
from data_preprocessing import clean_text
from feature_extraction import load_vectorizer

def predict(text, model_path="LogisticRegression_fake_news_model.joblib", vectorizer_path="tfidf_vectorizer.joblib"):
    # Clean the input text
    clean = clean_text(text)
    vect = load_vectorizer(vectorizer_path)
    X = vect.transform([clean])

    model = joblib.load(model_path)
    pred = model.predict(X)[0]
    return "FAKE NEWS" if pred == 1 else "REAL NEWS"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="News article text to classify")
    parser.add_argument("--model", default="LogisticRegression_fake_news_model.joblib", help="Path to trained model")
    parser.add_argument("--vectorizer", default="tfidf_vectorizer.joblib", help="Path to saved TF-IDF vectorizer")
    args = parser.parse_args()

    result = predict(args.text, args.model, args.vectorizer)
    print("[INFO] Prediction:", result)