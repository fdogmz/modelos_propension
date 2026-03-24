import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
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
THRESHOLD = 0.50
N_GROUPS = 5  # Quintiles

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
def fit_logistic_cv(X_train, y_train):
    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_

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
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    return {
        "ROC-AUC": roc_auc_score(y_test, y_score),
        "PR-AUC": average_precision_score(y_test, y_score),
        "Exhaustividad": recall_score(y_test, y_pred, zero_division=0),
        "Precisión": precision_score(y_test, y_pred, zero_division=0),
        "Exactitud": accuracy_score(y_test, y_pred),
        "y_score": y_score,
        "y_pred": y_pred,
    }

def build_quintile_table(y_true, y_score, n_groups=5):
    df_scores = pd.DataFrame({
        "real": np.array(y_true),
        "score": np.array(y_score)
    }).sort_values("score", ascending=False).reset_index(drop=True)

    df_scores["quintil"] = pd.qcut(
        df_scores.index + 1,
        q=n_groups,
        labels=[f"Q{i}" for i in range(1, n_groups + 1)]
    )

    total_positives = df_scores["real"].sum()
    total_obs = len(df_scores)

    summary = (
        df_scores.groupby("quintil", observed=False)
        .agg(
            clientes=("real", "size"),
            churners=("real", "sum"),
            score_promedio=("score", "mean")
        )
        .reset_index()
    )

    summary["clientes_acum"] = summary["clientes"].cumsum()
    summary["churners_acum"] = summary["churners"].cumsum()
    summary["pct_clientes_acum"] = summary["clientes_acum"] / total_obs
    summary["pct_churners_acum"] = summary["churners_acum"] / total_positives
    summary["lift"] = summary["pct_churners_acum"] / summary["pct_clientes_acum"]

    return summary

def plot_gain_curve(quintile_df):
    fig, ax = plt.subplots()

    x = np.concatenate([[0], quintile_df["pct_clientes_acum"].values])
    y = np.concatenate([[0], quintile_df["pct_churners_acum"].values])

    ax.plot(x, y, marker="o", label="Modelo")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Selección aleatoria")

    ax.set_xlabel("Proporción acumulada de clientes")
    ax.set_ylabel("Proporción acumulada de churn capturado")
    ax.set_title("Curva de ganancia acumulada")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()

    return fig

def plot_lift_curve(quintile_df):
    fig, ax = plt.subplots()

    x = quintile_df["pct_clientes_acum"].values
    y = quintile_df["lift"].values

    ax.plot(x, y, marker="o", label="Lift acumulado")
    ax.axhline(1.0, linestyle="--", label="Referencia aleatoria")

    ax.set_xlabel("Proporción acumulada de clientes")
    ax.set_ylabel("Lift")
    ax.set_title("Curva de lift")
    ax.set_xlim(0, 1)
    ax.legend()

    return fig

# --------------------------------------------------
# Página
# --------------------------------------------------
page_header(
    "8. Comparación de modelos y valor de negocio",
    "Comparación entre regresión logística y Random Forest, con lift y ganancia"
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
# Ajuste de modelos
# --------------------------------------------------
log_model = fit_logistic_cv(X_train_bal, y_train_bal)
rf_model = fit_rf_cv(X_train_bal, y_train_bal)

log_results = evaluate_model(log_model, X_test, y_test, threshold=THRESHOLD)
rf_results = evaluate_model(rf_model, X_test, y_test, threshold=THRESHOLD)

comparison_df = pd.DataFrame([
    {
        "Modelo": "Regresión logística",
        "ROC-AUC": log_results["ROC-AUC"],
        "PR-AUC": log_results["PR-AUC"],
        "Exhaustividad": log_results["Exhaustividad"],
        "Precisión": log_results["Precisión"],
        "Exactitud": log_results["Exactitud"],
    },
    {
        "Modelo": "Random Forest",
        "ROC-AUC": rf_results["ROC-AUC"],
        "PR-AUC": rf_results["PR-AUC"],
        "Exhaustividad": rf_results["Exhaustividad"],
        "Precisión": rf_results["Precisión"],
        "Exactitud": rf_results["Exactitud"],
    },
])

# --------------------------------------------------
# Comparación
# --------------------------------------------------
st.markdown("## Comparación entre modelos")

st.markdown(f"""
La siguiente tabla resume el desempeño de ambos modelos sobre el conjunto de prueba.
Las métricas **Exhaustividad**, **Precisión** y **Exactitud** se calcularon con un umbral común de **{THRESHOLD:.2f}**.
""")

st.dataframe(
    comparison_df.style.format({
        "ROC-AUC": "{:.4f}",
        "PR-AUC": "{:.4f}",
        "Exhaustividad": "{:.4f}",
        "Precisión": "{:.4f}",
        "Exactitud": "{:.4f}",
    }),
    use_container_width=True,
    hide_index=True
)

st.info("""
En este caso, Random Forest ofrece una mejora ligera en ROC-AUC, PR-AUC y exhaustividad, por lo que se adopta como modelo de referencia para el análisis de valor de negocio, especialmente en tareas de priorización de clientes en riesgo.
""")



# --------------------------------------------------
# Lift y ganancia
# --------------------------------------------------
st.markdown("## Lift y ganancia con Random Forest")

st.markdown("""
Para explorar el valor de negocio del modelo, se ordenan los clientes del conjunto de prueba
según su probabilidad estimada de abandono y se dividen en **quintiles**.
De este modo puede evaluarse qué proporción de clientes con mayor score concentra
una fracción importante de los churners observados.
""")

quintile_df = build_quintile_table(y_test, rf_results["y_score"], n_groups=N_GROUPS)

display_quintile_df = quintile_df.copy()
#display_quintile_df = quintile_df.copy()

display_quintile_df["score_promedio"] = display_quintile_df["score_promedio"].round(4)
display_quintile_df["pct_clientes_acum"] = display_quintile_df["pct_clientes_acum"].round(4)
display_quintile_df["pct_churners_acum"] = display_quintile_df["pct_churners_acum"].round(4)
display_quintile_df["lift"] = display_quintile_df["lift"].round(4)

styled_quintile_df = (
    display_quintile_df.style
    .format({
        "clientes": "{:,.0f}",
        "churners": "{:,.0f}",
        "score_promedio": "{:.4f}",
        "clientes_acum": "{:,.0f}",
        "churners_acum": "{:,.0f}",
        "pct_clientes_acum": "{:.1%}",
        "pct_churners_acum": "{:.1%}",
        "lift": "{:.3f}",
    })
    .background_gradient(subset=["churners"], cmap="Blues")
)


st.markdown("### Tabla por quintiles")
#st.dataframe(display_quintile_df, use_container_width=True, hide_index=True)
st.dataframe(styled_quintile_df, use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Curva de ganancia acumulada")
    fig_gain = plot_gain_curve(quintile_df)
    st.pyplot(fig_gain)

with c2:
    st.markdown("### Curva de lift")
    fig_lift = plot_lift_curve(quintile_df)
    st.pyplot(fig_lift)

st.markdown("## Interpretación de negocio")

top_quintile_capture = quintile_df.iloc[0]["pct_churners_acum"]
top_two_quintiles_capture = quintile_df.iloc[1]["pct_churners_acum"]

st.success(
    f"El primer quintil concentra aproximadamente el **{top_quintile_capture:.2%}** de los churners, "
    f"mientras que los dos primeros quintiles acumulan alrededor del **{top_two_quintiles_capture:.2%}**. "
    "Esto muestra que el modelo permite focalizar acciones de retención sobre una fracción relativamente "
    "pequeña de clientes, capturando una parte desproporcionada del riesgo de abandono."
)