import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def generate_synthetic_propensity_data(
    n_samples: int = 2000,
    imbalance_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Genera un dataset sintetico binario para ejemplos de propension."""
    imbalance_ratio = float(np.clip(imbalance_ratio, 0.01, 0.99))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=6,
        n_redundant=3,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[1.0 - imbalance_ratio, imbalance_ratio],
        class_sep=1.0,
        flip_y=0.01,
        random_state=random_state,
    )

    feature_names = [f"feature_{idx:02d}" for idx in range(1, X.shape[1] + 1)]
    data = pd.DataFrame(X, columns=feature_names)
    data["target"] = y.astype(int)
    return data


def get_feature_target(
    data: pd.DataFrame,
    target_col: str = "target",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variables predictoras y objetivo."""
    X = data.drop(columns=[target_col])
    y = data[target_col]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Realiza particion estratificada train/test."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
