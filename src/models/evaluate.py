from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):

    predictions = model.predict(X_test)

    print("\n")
    print("=" * 50)
    print(model_name)
    print("=" * 50)

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            predictions
        )
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )