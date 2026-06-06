import pandas as pd

from sklearn.model_selection import train_test_split

from src.preprocessing.cleaner import clean_text

from src.feature_engineering.tfidf_vectorizer import (
    train_vectorizer
)

from src.models.logistic_regression import (
    train_lr
)

from src.models.naive_bayes import (
    train_nb
)

from src.models.evaluate import (
    evaluate_model
)

from src.utils.config import (
    RAW_DATA_DIR
)

import os
import joblib


# ======================
# Load Dataset
# ======================

fake_path = os.path.join(
    RAW_DATA_DIR,
    "Fake.csv"
)

true_path = os.path.join(
    RAW_DATA_DIR,
    "True.csv"
)

print("Loading datasets...")

fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(true_path)

# ======================
# Create Labels
# ======================

fake_df["label"] = 0
real_df["label"] = 1

# ======================
# Merge Dataset
# ======================

news_df = pd.concat(
    [fake_df, real_df],
    ignore_index=True
)

news_df = news_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("Dataset Shape:")
print(news_df.shape)

# ======================
# Clean Text
# ======================

print("Cleaning text...")

news_df["combined_text"] = (
    news_df["title"].fillna("")
    + " "
    + news_df["text"].fillna("")
)

news_df["clean_text"] = news_df["combined_text"].apply(
    clean_text
)

# ======================
# Features and Labels
# ======================

X = news_df["clean_text"]

y = news_df["label"]

# ======================
# Train Test Split
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Size:", len(X_train))
print("Test Size:", len(X_test))

# ======================
# TF-IDF
# ======================

print("\nCreating TF-IDF vectors...")

X_train_vec, vectorizer = train_vectorizer(
    X_train
)

X_test_vec = vectorizer.transform(
    X_test
)

# ======================
# Logistic Regression
# ======================

print("\nTraining Logistic Regression...")

lr_model = train_lr(
    X_train_vec,
    y_train
)

evaluate_model(
    lr_model,
    X_test_vec,
    y_test,
    "Logistic Regression"
)

# ======================
# Naive Bayes
# ======================

print("\nTraining Naive Bayes...")

nb_model = train_nb(
    X_train_vec,
    y_train
)

evaluate_model(
    nb_model,
    X_test_vec,
    y_test,
    "Naive Bayes"
)

print("\nTraining Completed Successfully!")

print("\nSaved Models:")
print("- tfidf.pkl")
print("- logistic_model.pkl")
print("- nb_model.pkl")
