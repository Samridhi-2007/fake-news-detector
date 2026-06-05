import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "trained_models"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

RAW_DATA_DIR = os.path.join(
    DATA_DIR,
    "raw"
)

PROCESSED_DATA_DIR = os.path.join(
    DATA_DIR,
    "processed"
)

TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "tfidf.pkl"
)

LR_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "logistic_model.pkl"
)

NB_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "nb_model.pkl"
)