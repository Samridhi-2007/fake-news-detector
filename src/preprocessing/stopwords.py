from nltk.corpus import stopwords

STOPWORDS = set(
    stopwords.words("english")
)

CUSTOM_STOPWORDS = {
    "said",
    "would",
    "could",
    "also",
    "one",
    "two",
    "new"
}

STOPWORDS.update(
    CUSTOM_STOPWORDS
)