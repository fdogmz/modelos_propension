import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)

from src.ui.components import page_header

# --------------------------------------------------
# Configuración
# --------------------------------------------------
DATA_PATH = Path("data/processed/churn_case_analysis_base.parquet")
TARGET_COL = "CHURN"   # Cambia si tu variable objetivo tiene otro nombre
CATEGORICAL_COLS = ["GENDER", "MARITAL_STATUS"]
TEST_SIZE = 0.30
RANDOM_STATE = 42
N_SPLITS = 5

st.set_page_config(layout="wide")

# --------------------------------------------------
# Utilidades
# --------------------------------------------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data
def prepare_train_test(df: pd.DataFrame, target_col: str, categorical_cols: list[str]):
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    cat_cols_present = [col for col in categorical_cols if col in X.columns]
    X_encoded = pd.get_dummies(X, columns=cat_cols_present, drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    class_counts = train_df[target_col].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    minority_size = class_counts.min()

    train_minority = train_df[train_df[target_col] == minority_class]
    train_majority = train_df[train_df[target_col] == majority_class].sample(
        n=minority_size,
        random_state=RANDOM_STATE
    )

    train_balanced = pd.concat([train_minority, train_majority], axis=0).sample(
        frac=1,
        random_state=RANDOM_STATE
    )

    X_train_bal = train_balanced.drop(columns=[target_col])
    y_train_bal = train_balanced[target_col]

    return X_train_bal, X_test, y_train_bal, y_test, X_encoded.columns.tolist()

@st.cache_data
def fit_rf_cv(X_train, y_train):
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 8, None],
        "min_samples_leaf": [1, 5, 10],
    }

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="average_precision",   # PR-AUC
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    cv_results = pd.DataFrame(grid.cv_results_)[
        [
            "param_n_estimators",
            "param_max_depth",
            "param_min_samples_leaf",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "rank_test_score",
        ]
    ].rename(
        columns={
            "param_n_estimators": "n_estimators",
            "param_max_depth": "max_depth",
            "param_min_samples_leaf": "min_samples_leaf",
            "mean_test_score": "mean_pr_auc_cv",
            "std_test_score": "std_pr_auc_cv",
            "mean_train_score": "mean_train_pr_auc",
            "rank_test_score": "rank",
        }
    ).sort_values(["rank", "mean_pr_auc_cv"], ascending=[True, False])

    best_model = grid.best_estimator_

    return best_model, cv_results, grid.best_params_, grid.best_score_

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    positive_rate = float(np.mean(y_test))

    metrics = {
        "exactitud": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "exhaustividad": recall_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_score),
        "roc_auc": roc_auc_score(y_test, y_score),
        "tasa_base": positive_rate,
    }

    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_score, y_pred

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Matriz de confusión en prueba")
    return fig

def plot_precision_recall_curve_standard(y_true, y_score, threshold_selected=None):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    base_rate = np.mean(y_true)

    fig, ax = plt.subplots()

    ax.plot(recall, precision, linewidth=2, label="Curva Precision–Recall")

    ax.axhline(
        y=base_rate,
        linestyle="--",
        linewidth=1.5,
        label=f"Referencia (tasa base = {base_rate:.4f})"
    )

    if threshold_selected is not None:
        y_pred_thr = (y_score >= threshold_selected).astype(int)
        precision_thr = precision_score(y_true, y_pred_thr, zero_division=0)
        recall_thr = recall_score(y_true, y_pred_thr, zero_division=0)

        ax.scatter(
            recall_thr,
            precision_thr,
            s=90,
            label=f"Umbral = {threshold_selected:.2f}"
        )

    ax.set_xlabel("Exhaustividad")
    ax.set_ylabel("Precisión")
    ax.set_title("Curva Precision–Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()

    return fig

def plot_cv_results_rf(cv_results: pd.DataFrame, top_n=8):
    top = cv_results.head(top_n).copy()
    top["config"] = (
        "n=" + top["n_estimators"].astype(str)
        + ", depth=" + top["max_depth"].astype(str)
        + ", leaf=" + top["min_samples_leaf"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(top["config"][::-1], top["mean_pr_auc_cv"][::-1])
    ax.set_xlabel("PR-AUC media en CV")
    ax.set_ylabel("Configuración")
    ax.set_title("Mejores configuraciones en validación cruzada")
    return fig

def get_feature_importance_df(model, feature_names):
    importances = pd.DataFrame({
        "variable": feature_names,
        "importancia": model.feature_importances_
    }).sort_values("importancia", ascending=False)
    return importances

def plot_feature_importance(importances_df, top_n=10):
    top = importances_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["variable"][::-1], top["importancia"][::-1])
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")
    ax.set_title(f"Top {top_n} variables más importantes")
    return fig

# --------------------------------------------------
# Página
# --------------------------------------------------
page_header(
    "7. Modelado con Random Forest",
    "Ajuste de hiperparámetros mediante validación cruzada y evaluación en prueba"
)

if not DATA_PATH.exists():
    st.error(f"No se encontró el archivo: {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)

if TARGET_COL not in df.columns:
    st.error(
        f"La variable objetivo '{TARGET_COL}' no existe en el dataset. "
        f"Columnas disponibles: {list(df.columns)}"
    )
    st.stop()

X_train_bal, X_test, y_train_bal, y_test, encoded_cols = prepare_train_test(
    df, TARGET_COL, CATEGORICAL_COLS
)

# --------------------------------------------------
# Introducción
# --------------------------------------------------
st.markdown("""
**Random Forest** es un método de ensamble basado en múltiples árboles de decisión.
En este caso se utiliza para estimar la probabilidad de abandono a partir de las variables históricas
del cliente.
""")

st.info("""
En esta página, el modelo se ajusta sobre el **conjunto de entrenamiento balanceado**.
La selección de hiperparámetros se realiza mediante **validación cruzada K-fold**,
y la evaluación final se reporta sobre el **conjunto de prueba original**.
""")

# --------------------------------------------------
# Hiperparámetros
# --------------------------------------------------
st.markdown("## Hiperparámetros ajustados")

st.markdown("""
En Random Forest se ajustaron tres hiperparámetros principales:

- **n_estimators**: número de árboles del ensamble;
- **max_depth**: profundidad máxima de cada árbol;
- **min_samples_leaf**: tamaño mínimo permitido en una hoja terminal.
""")

st.latex(r"n\_estimators \in \{100,\ 200\}")
st.latex(r"max\_depth \in \{4,\ 8,\ \text{None}\}")
st.latex(r"min\_samples\_leaf \in \{1,\ 5,\ 10\}")

# --------------------------------------------------
# Ajuste con CV
# --------------------------------------------------
st.markdown("## Ajuste mediante validación cruzada")

best_model, cv_results, best_params, best_score = fit_rf_cv(X_train_bal, y_train_bal)

left, right = st.columns([1, 1.2])

with left:
    st.markdown("""
Se utilizó **Stratified K-Fold** con 5 particiones sobre el conjunto de entrenamiento balanceado.
La métrica de selección fue **PR-AUC**, adecuada para problemas con clases desbalanceadas.
""")
    st.metric("Mejor n_estimators", f"{best_params['n_estimators']}")
    st.metric("Mejor max_depth", f"{best_params['max_depth']}")
    st.metric("Mejor min_samples_leaf", f"{best_params['min_samples_leaf']}")
    st.metric("Mejor PR-AUC media en CV", f"{best_score:.4f}")

with right:
    st.markdown("### Resultados de validación cruzada")
    st.dataframe(cv_results, use_container_width=True, hide_index=True)

st.markdown("### Mejores configuraciones en validación cruzada")
fig_cv = plot_cv_results_rf(cv_results, top_n=8)
st.pyplot(fig_cv)

# --------------------------------------------------
# Evaluación en test
# --------------------------------------------------
st.markdown("## Generalización sobre el conjunto de prueba")

metrics, cm, y_score, y_pred = evaluate_model(best_model, X_test, y_test, threshold=0.5)

st.markdown("### Métricas independientes del umbral")
a1, a2 = st.columns(2)
a1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
a2.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")

st.markdown("""
Estas métricas se calcularon sobre el **conjunto de prueba con distribución original**.
De este modo, la evaluación refleja mejor las condiciones del problema real.
""")

st.info(
    f"La tasa base de churn en el conjunto de prueba es {metrics['tasa_base']:.4f}. "
    "En problemas desbalanceados, el ROC-AUC y el PR-AUC pueden diferir de forma importante: "
    "el ROC-AUC mide la capacidad global de discriminación, mientras que el PR-AUC pone más énfasis "
    "en el desempeño sobre la clase positiva."
)

st.markdown("### Explora el efecto del umbral de decisión")

threshold = st.slider(
    "Selecciona el umbral de probabilidad para clasificar abandono",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.01,
    key="rf_threshold"
)

metrics, cm, y_score, y_pred = evaluate_model(best_model, X_test, y_test, threshold=threshold)

st.markdown(f"### Métricas dependientes del umbral (umbral = {threshold:.2f})")
m1, m2, m3 = st.columns(3)
m1.metric("Exhaustividad", f"{metrics['exhaustividad']:.4f}")
m2.metric("Precisión", f"{metrics['precision']:.4f}")
m3.metric("Exactitud", f"{metrics['exactitud']:.4f}")

st.caption(
    "La matriz de confusión y métricas como precisión, exhaustividad y exactitud "
    "sí dependen del umbral elegido."
)

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("### Matriz de confusión")
    fig_cm = plot_confusion_matrix(cm)
    st.pyplot(fig_cm)

with c2:
    results_preview = pd.DataFrame({
        "probabilidad_abandono": y_score,
        "prediccion": y_pred,
        "real": y_test.values if hasattr(y_test, "values") else y_test
    }).sort_values("probabilidad_abandono", ascending=False).head(10)

    results_preview["supera_umbral"] = results_preview["probabilidad_abandono"] >= threshold
    results_preview = results_preview[
        ["probabilidad_abandono", "supera_umbral", "prediccion", "real"]
    ]

    st.markdown("### Ejemplo de probabilidades y decisión")
    st.dataframe(results_preview, use_container_width=True, hide_index=True)

st.markdown("### Curva Precision–Recall")
fig_pr = plot_precision_recall_curve_standard(y_test, y_score, threshold)
st.pyplot(fig_pr)

st.caption(
    "La línea punteada representa la tasa base de churn en el conjunto de prueba. "
    "La curva muestra el compromiso entre precisión y exhaustividad para distintos umbrales. "
    "El punto resaltado corresponde al umbral actualmente seleccionado."
)

# --------------------------------------------------
# Importancia de variables
# --------------------------------------------------
st.markdown("## Importancia de variables")

importance_df = get_feature_importance_df(best_model, X_train_bal.columns.tolist())

i1, i2 = st.columns([1.1, 1])

with i1:
    fig_imp = plot_feature_importance(importance_df, top_n=10)
    st.pyplot(fig_imp)

with i2:
    st.markdown("### Variables más importantes")
    st.dataframe(importance_df.head(10), use_container_width=True, hide_index=True)

# --------------------------------------------------
# Interpretación breve
# --------------------------------------------------
st.markdown("## Lectura de resultados")

st.markdown("""
Random Forest ofrece un enfoque más flexible que la regresión logística:

- puede capturar relaciones no lineales;
- incorpora interacciones entre variables de manera implícita;
- y permite analizar la importancia relativa de las variables predictoras.

En problemas de churn, esto puede traducirse en mejores probabilidades estimadas y en una
mayor capacidad para identificar clientes en riesgo, aunque a costa de una menor interpretabilidad
estructural que la de un modelo lineal.
""")

st.success("""
Con esta página ya es posible comparar un modelo lineal con un método de ensamble bajo
un mismo esquema de evaluación. El siguiente paso natural puede ser contrastar explícitamente
los resultados de ambos modelos.
""")