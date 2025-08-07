import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from data_preprocessing import preprocess_data
from feature_extraction import extract_features

def main():
    df = preprocess_data("data/news.csv")
    X = extract_features(df["content"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Try multiple classifiers
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "NaiveBayes": MultinomialNB()
    }

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(classification_report(y_test, preds))
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        joblib.dump(model, f"{name}_fake_news_model.joblib")
        print(f"Saved: {name}_fake_news_model.joblib")

if __name__ == "__main__":
    main()