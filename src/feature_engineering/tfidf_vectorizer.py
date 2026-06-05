from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from src.utils.config import TFIDF_PATH


def train_vectorizer(texts):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(texts)

    joblib.dump(
        vectorizer,
        TFIDF_PATH
    )

    return X, vectorizer