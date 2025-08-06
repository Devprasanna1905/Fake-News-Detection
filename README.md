# Fake News Detection Using Machine Learning

This project implements a machine learning pipeline to detect fake news articles by leveraging advanced Natural Language Processing (NLP) techniques, feature engineering with TF-IDF, and classification algorithms. It demonstrates practical applications of text data cleansing, vectorization, and model building in Python.

## Features

- Cleans and preprocesses news article data.
- Extracts informative features using TF-IDF vectorization.
- Trains and evaluates multiple classification algorithms (Logistic Regression, Naive Bayes, Random Forest, etc.).
- Provides accuracy metrics and confusion matrix for performance assessment.
- Includes a simple command-line or notebook interface for predictions on new articles.

## Setup and Installation

1. **Clone this repository:**
    ```
    git clone https://github.com/yourusername/fake-news-detection.git
    cd fake-news-detection
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Prepare the dataset:**
    - Add your news articles data in `data/news.csv`, containing columns such as `title`, `text`, and `label` (where label is 1 for fake, 0 for real).

4. **Run preprocessing & model training:**
    ```
    python src/train_model.py
    ```

5. **Make predictions on new samples (optional):**
    ```
    python src/predict.py --text "Your news article text here"
    ```

## Methods

- **NLP Preprocessing:** Tokenization, stopword removal, stemming/lemmatization.
- **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors suitable for ML models.
- **Classification Algorithms:** Built and tested several models including Logistic Regression, Naive Bayes, Random Forest, and more.
- **Evaluation Metrics:** Accuracy, precision, recall, F1 score, and confusion matrix.

## Endpoints / Usage

- Model training can be invoked with `train_model.py`.
- Predictions for new articles can be made via the `predict.py` script or a Jupyter Notebook interface.
- Optional: Add a Flask or FastAPI web interface for API-based predictions.

## Example Usage

$ python src/predict.py --text "Government confirms new policies for economic growth."
[INFO] Prediction: REAL NEWS

$ python src/predict.py --text "Aliens landed and signed trade agreement in New York."
[INFO] Prediction: FAKE NEWS
