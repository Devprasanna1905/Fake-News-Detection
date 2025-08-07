from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def extract_features(train_texts, vectorizer_path="tfidf_vectorizer.joblib"):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(train_texts)
    joblib.dump(vectorizer, vectorizer_path)
    return X

def load_vectorizer(vectorizer_path="tfidf_vectorizer.joblib"):
    return joblib.load(vectorizer_path)