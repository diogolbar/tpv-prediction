import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import pandas as pd
from lightgbm import LGBMClassifier


def create_classification_dashboard(
    classifier: LGBMClassifier, X_val: pd.DataFrame, y_val: pd.Series
) -> plt.Figure:
    """
    Create a dashboard for monitoring a multi-class classification
    machine learning classifier.

    Args:
        classifier: Trained LGBMClassifier classifier.
        X_val: DataFrame containing the validation features.
        y_val: Series containing the true values for the predicted classes.

    Returns:
        plt.Figure: A figure containing the dashboard elements.
    """
    y_val = y_val.astype("category")

    # Make predictions
    predictions = classifier.predict(X_val)
    predictions_proba = classifier.predict_proba(X_val)

    # Confusion Matrix
    cm = confusion_matrix(y_val, predictions)
    fig, axs = plt.subplots(
        2, 2 + predictions_proba.shape[1], figsize=(20, 15))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[0, 0])
    axs[0, 0].set_title("Confusion Matrix")
    axs[0, 0].set_xlabel("Predicted Label")
    axs[0, 0].set_ylabel("True Label")

    # Metrics
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, average="macro")
    recall = recall_score(y_val, predictions, average="macro")
    f1 = f1_score(y_val, predictions, average="macro")

    metrics_data = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }
    metrics_df = pd.DataFrame(metrics_data, index=[0])
    axs[0, 1].axis("off")
    axs[0, 1].table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        cellLoc="center",
        loc="center",
    )

    # Predicted Probabilities
    if predictions_proba is not None:
        # Add class names to the predictions_proba DataFrame
        class_names = [f"Class {i}" for i in range(predictions_proba.shape[1])]
        predictions_proba_df = pd.DataFrame(
            predictions_proba, columns=class_names)

        for i in range(predictions_proba_df.shape[1]):
            sns.histplot(
                predictions_proba_df.iloc[:, i], ax=axs[1, i + 2], kde=True)
            axs[1, i +
                2].set_title(f"Class {i} Predicted Probabilities Distribution")
            axs[1, i + 2].set_xlabel("Probability")
            axs[1, i + 2].set_ylabel("Frequency")

    plt.tight_layout()
    return fig
