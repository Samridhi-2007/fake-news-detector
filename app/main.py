from fastapi import FastAPI

from app.schemas import NewsRequest
from src.predict import predict_news

app = FastAPI(
    title="Fake News Detector API"
)

@app.post("/predict")
def predict(request: NewsRequest):

    return predict_news(
        request.text
    )