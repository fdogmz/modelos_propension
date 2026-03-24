from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_logistic_regression(X_train, y_train, C: float = 1.0, random_state: int = 42):
    """Entrena una regresion logistica dentro de un pipeline basico."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model
