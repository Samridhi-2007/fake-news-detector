from sklearn.naive_bayes import MultinomialNB
import joblib

from src.utils.config import NB_MODEL_PATH


def train_nb(X_train, y_train):

    model = MultinomialNB()

    model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        model,
        NB_MODEL_PATH
    )

    return model