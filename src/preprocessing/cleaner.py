import re

from src.preprocessing.tokenizer import tokenize
from src.preprocessing.stopwords import STOPWORDS


def clean_text(text):
    """
    Complete text preprocessing pipeline.
    """

    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # remove html tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # remove punctuation and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # tokenize
    tokens = tokenize(text)

    # remove stopwords
    tokens = [
        token
        for token in tokens
        if token not in STOPWORDS
    ]

    cleaned_text = " ".join(tokens)

    return cleaned_text