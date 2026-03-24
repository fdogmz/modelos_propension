import pandas as pd


def safe_divide(numerator: float, denominator: float) -> float:
    """Division segura evitando errores por cero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def format_percent(value: float, decimals: int = 2) -> str:
    """Formatea un valor numerico como porcentaje."""
    return f"{value * 100:.{decimals}f}%"


def summarize_target(y: pd.Series) -> pd.DataFrame:
    """Crea tabla simple de conteo y proporcion por clase."""
    counts = y.value_counts(dropna=False).sort_index()
    total = counts.sum()
    summary = pd.DataFrame({"count": counts})
    summary["ratio"] = summary["count"] / total
    return summary
