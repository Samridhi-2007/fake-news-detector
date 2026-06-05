from src.predict import predict_news

sample_news = """
Scientists have developed a new vaccine
that showed promising results in trials.
"""

result = predict_news(
    sample_news
)

print(result)