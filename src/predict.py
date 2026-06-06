import joblib

from src.preprocessing.cleaner import clean_text

from src.utils.config import (
    LR_MODEL_PATH,
    TFIDF_PATH
)

FAKE_DECISION_THRESHOLD = 0.75

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

    probability = model.predict_proba(
        vector
    )[0]

    fake_probability = probability[0]
    real_probability = probability[1]

    if fake_probability >= FAKE_DECISION_THRESHOLD:
        label = "Fake News"
        confidence = fake_probability * 100
    else:
        label = "Real News"
        confidence = real_probability * 100

    return {
        "prediction": label,
        "confidence": float(round(
            confidence,
            2
        )),
        "fake_probability": float(round(
            fake_probability * 100,
            2
        )),
        "real_probability": float(round(
            real_probability * 100,
            2
        ))
    }
