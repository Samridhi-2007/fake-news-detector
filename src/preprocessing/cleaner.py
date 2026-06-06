import re

from src.preprocessing.tokenizer import tokenize


def strip_source_boilerplate(text):
    """
    Remove dataset-specific wire-service prefixes before cleaning.
    """

    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"^\s*[A-Z][A-Z\s\.\-]+\(Reuters\)\s*-\s*",
        "",
        text
    )

    text = re.sub(
        r"^\s*\(Reuters\)\s*-\s*",
        "",
        text
    )

    return text


def clean_text(text):
    """
    Complete text preprocessing pipeline.
    """

    if not isinstance(text, str):
        return ""

    text = strip_source_boilerplate(text)

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

    cleaned_text = " ".join(tokens)

    return cleaned_text
