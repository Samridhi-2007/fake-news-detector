from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def tokenize(text):
    """
    Tokenize and lemmatize text.
    """

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]

    return tokens