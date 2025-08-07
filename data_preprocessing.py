import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download("stopwords")

def clean_text(text):
    # Lowercase, remove punctuation, remove stopwords, and stem
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df["content"] = df["title"].fillna('') + " " + df["text"].fillna('')
    df["content"] = df["content"].apply(clean_text)
    return df[["content", "label"]]