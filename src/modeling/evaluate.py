import numpy as np

from src.utils.metrics import classification_metrics, confusion_df


def evaluate_binary_classifier(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Evalua un clasificador binario y retorna metricas utiles."""
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)

    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    return {
        "metrics": classification_metrics(y_test, y_pred, y_prob=y_prob),
        "confusion_matrix": confusion_df(y_test, y_pred),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }
