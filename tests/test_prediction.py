from src.predict import predict_news


def test_short_factual_news_is_not_forced_to_fake():
    result = predict_news(
        "Scientists have developed a new vaccine that showed promising results in trials."
    )

    assert result["prediction"] == "Real News"
    assert "fake_probability" in result
    assert "real_probability" in result


def test_short_finance_policy_news_is_not_forced_to_fake():
    result = predict_news(
        "The Reserve Bank announced that interest rates will remain unchanged after its latest policy review."
    )

    assert result["prediction"] == "Real News"


def test_sensational_claim_still_predicts_fake():
    result = predict_news(
        "Breaking secret miracle cure doctors do not want you to know has been revealed."
    )

    assert result["prediction"] == "Fake News"
