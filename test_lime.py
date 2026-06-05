from src.predict import predict_news
from src.explainability.lime_explainer import explain_news

news = """
Scientists have developed a new vaccine
that showed promising results in trials.
"""

prediction = predict_news(news)

print("Prediction:")
print(prediction)

print("\nImportant Words:")

for item in explain_news(news):
    print(item)