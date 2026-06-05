import joblib

from src.preprocessing.cleaner import clean_text

from src.utils.config import (
    LR_MODEL_PATH,
    TFIDF_PATH
)

# Load trained model
model = joblib.load(
    LR_MODEL_PATH
)

# Load trained vectorizer
vectorizer = joblib.load(
    TFIDF_PATH
)


def predict_news(news_text):

    cleaned_text = clean_text(
        news_text
    )

    vector = vectorizer.transform(
        [cleaned_text]
    )

    prediction = model.predict(
        vector
    )[0]

    probability = model.predict_proba(
        vector
    )[0]

    confidence = max(
        probability
    ) * 100

    label = (
        "Real News"
        if prediction == 1
        else "Fake News"
    )

    return {
        "prediction": label,
        "confidence": round(
            confidence,
            2
        )
    }