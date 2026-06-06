import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)
sys.path.insert(0, ROOT_DIR)

import streamlit as st
import importlib

import src.predict as prediction_module

prediction_module = importlib.reload(
    prediction_module
)

predict_news = prediction_module.predict_news

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

        if (
            "fake_probability" in result
            and "real_probability" in result
        ):
            st.write(
                f"Fake score: {result['fake_probability']}%"
            )
            st.write(
                f"Real score: {result['real_probability']}%"
            )
