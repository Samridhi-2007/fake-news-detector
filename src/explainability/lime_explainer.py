from lime.lime_text import LimeTextExplainer

from src.predict import (
    model,
    vectorizer
)

# Class names used by the model
class_names = [
    "Fake News",
    "Real News"
]

explainer = LimeTextExplainer(
    class_names=class_names
)


def predict_proba(texts):
    """
    LIME expects a list of raw texts
    and returns prediction probabilities.
    """

    cleaned_texts = [
        text for text in texts
    ]

    vectors = vectorizer.transform(
        cleaned_texts
    )

    return model.predict_proba(
        vectors
    )


def explain_news(news_text, num_features=5):

    explanation = explainer.explain_instance(
        news_text,
        predict_proba,
        num_features=num_features
    )

    important_words = []

    for word, score in explanation.as_list():
        important_words.append(
            {
                "word": word,
                "importance": round(
                    score,
                    4
                )
            }
        )

    return important_words