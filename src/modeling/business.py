import numpy as np


def expected_profit(
    y_true,
    y_pred,
    benefit_tp: float = 100.0,
    cost_fp: float = 15.0,
    cost_fn: float = 40.0,
    cost_tn: float = 0.0,
) -> float:
    """Calcula utilidad esperada basada en una matriz de costos/beneficios."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    return tp * benefit_tp - fp * cost_fp - fn * cost_fn - tn * cost_tn


def lift_at_k(y_true, y_score, k: float = 0.2) -> float:
    """Calcula lift para el top k de observaciones con mayor score."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) == 0:
        return 0.0

    k = float(np.clip(k, 0.01, 1.0))
    cutoff = max(1, int(len(y_true) * k))

    ranked_idx = np.argsort(-y_score)
    top_true = y_true[ranked_idx][:cutoff]

    top_rate = top_true.mean() if len(top_true) else 0.0
    base_rate = y_true.mean() if len(y_true) else 0.0

    if base_rate == 0:
        return 0.0

    return float(top_rate / base_rate)
