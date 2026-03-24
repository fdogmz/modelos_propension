import matplotlib.pyplot as plt
import pandas as pd


def plot_class_distribution(y, title: str = "Distribucion de clases"):
    """Grafica barras simples para visualizar el desbalance de clases."""
    series = pd.Series(y)
    counts = series.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel("Clase")
    ax.set_ylabel("Cantidad")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    return fig
