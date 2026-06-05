import streamlit as st

from src.predict import predict_news
st.info(
    "This model predicts based on learned patterns from training data and should not be considered a definitive fact-checking system."
)
st.title("📰 Fake News Detector")

news = st.text_area(
    "Paste News Article"
)

if st.button("Analyze"):

    if news.strip():

        result = predict_news(news)

        st.success(
            f"Prediction: {result['prediction']}"
        )

        st.write(
            f"Confidence: {result['confidence']}%"
        )