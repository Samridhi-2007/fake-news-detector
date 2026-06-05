from sklearn.linear_model import LogisticRegression
import joblib

from src.utils.config import LR_MODEL_PATH


def train_lr(X_train, y_train):

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        model,
        LR_MODEL_PATH
    )

    return model