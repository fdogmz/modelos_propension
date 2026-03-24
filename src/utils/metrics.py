import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true,
    y_pred,
    y_prob=None,
) -> dict[str, float]:
    """Calcula metricas principales de clasificacion binaria."""
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        results["roc_auc"] = roc_auc_score(y_true, y_prob)

    return results


def confusion_df(y_true, y_pred) -> pd.DataFrame:
    """Devuelve la matriz de confusion en formato tabular."""
    matrix = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
