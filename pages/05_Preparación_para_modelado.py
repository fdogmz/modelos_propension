import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.ui.components import page_header

# -----------------------------
# Configuración
# -----------------------------
DATA_PATH = Path("data/processed/churn_case_analysis_base.parquet")
TARGET_COL = "CHURN"
CATEGORICAL_COLS = ["GENDER", "MARITAL_STATUS"]
TEST_SIZE = 0.30
RANDOM_STATE = 42

st.set_page_config(layout="wide")

# -----------------------------
# Utilidades
# -----------------------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data
def prepare_datasets(df: pd.DataFrame, target_col: str, categorical_cols: list[str]):
    # Separación X / y
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # Recodificación de variables categóricas
    cat_cols_present = [col for col in categorical_cols if col in X.columns]
    X_encoded = pd.get_dummies(X, columns=cat_cols_present, drop_first=False)

    # Partición estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Balanceo por submuestreo de la clase mayoritaria SOLO en train
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

    return {
        "X_encoded": X_encoded,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_bal": X_train_bal,
        "y_train_bal": y_train_bal,
        "encoded_columns": X_encoded.columns.tolist(),
        "class_counts_full": y.value_counts().sort_index(),
        "class_counts_train": y_train.value_counts().sort_index(),
        "class_counts_test": y_test.value_counts().sort_index(),
        "class_counts_train_bal": y_train_bal.value_counts().sort_index(),
    }

def build_distribution_df(results: dict, target_col: str) -> pd.DataFrame:
    parts = []
    mapping = {
        "Base completa": results["class_counts_full"],
        "Train original": results["class_counts_train"],
        "Test": results["class_counts_test"],
        "Train balanceado": results["class_counts_train_bal"],
    }

    for subset_name, counts in mapping.items():
        tmp = counts.rename_axis(target_col).reset_index(name="frecuencia")
        tmp["subset"] = subset_name
        tmp["porcentaje"] = 100 * tmp["frecuencia"] / tmp["frecuencia"].sum()
        parts.append(tmp)

    return pd.concat(parts, ignore_index=True)

def plot_class_distributions(dist_df: pd.DataFrame, target_col: str):
    subsets = dist_df["subset"].unique().tolist()

    fig, axes = plt.subplots(1, len(subsets), figsize=(4 * len(subsets), 4))
    if len(subsets) == 1:
        axes = [axes]

    for ax, subset in zip(axes, subsets):
        tmp = dist_df[dist_df["subset"] == subset]
        ax.bar(tmp[target_col].astype(str), tmp["frecuencia"])
        ax.set_title(subset)
        ax.set_xlabel(target_col)
        ax.set_ylabel("Frecuencia")

    plt.tight_layout()
    return fig

def build_size_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Conjunto": [
                "Base completa",
                "Entrenamiento original",
                "Prueba",
                "Entrenamiento balanceado",
            ],
            "Registros": [
                len(results["y"]),
                len(results["y_train"]),
                len(results["y_test"]),
                len(results["y_train_bal"]),
            ],
        }
    )

# -----------------------------
# Página
# -----------------------------
page_header(
    "5. Preparación de los datos para modelado",
    "Recodificación, partición estratificada y balanceo del conjunto de entrenamiento"
)

if not DATA_PATH.exists():
    st.error(f"No se encontró el archivo: {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)

if TARGET_COL not in df.columns:
    st.error(
        f"La variable objetivo '{TARGET_COL}' no está en el dataset. "
        f"Columnas disponibles: {list(df.columns)}"
    )
    st.stop()

results = prepare_datasets(df, TARGET_COL, CATEGORICAL_COLS)
dist_df = build_distribution_df(results, TARGET_COL)
size_df = build_size_table(results)

# -----------------------------
# Introducción
# -----------------------------
st.markdown("""
Antes de entrenar un modelo, la base analítica debe transformarse en un conjunto adecuado
para aprendizaje y evaluación. En este caso, la preparación incluyó tres pasos principales:

1. **recodificación de variables categóricas**;
2. **separación en conjuntos de entrenamiento y prueba mediante partición estratificada**;
3. **balanceo del conjunto de entrenamiento por submuestreo de la clase mayoritaria**.
""")

st.info("""
En este caso, la partición se realizó de forma **estratificada** y el balanceo se aplicó
**únicamente al conjunto de entrenamiento**. El conjunto de prueba conservó la distribución
original de clases para evaluar el modelo en condiciones más cercanas al problema real.
""")

# -----------------------------
# Recodificación
# -----------------------------
st.markdown("## Recodificación de variables categóricas")

categorical_present = [col for col in CATEGORICAL_COLS if col in df.columns]

if categorical_present:
    c1, c2 = st.columns([1, 1])

    st.markdown("### Variables categóricas identificadas")
    st.write(", ".join(categorical_present))

    c1, c2 = st.columns(2)

    with c1:
        example_before = df[categorical_present].head(8)
        st.markdown("### Ejemplo antes de recodificar")
        st.dataframe(example_before, use_container_width=True)

    with c2:
        encoded_example_cols = [
            col for col in results["encoded_columns"]
            if any(col.startswith(f"{cat}_") or col == cat for cat in categorical_present)
        ]
        encoded_example = results["X_encoded"][encoded_example_cols].head(8)

        st.markdown("### Ejemplo después de recodificar")
        st.dataframe(encoded_example, use_container_width=True)
        st.caption("""
        Las variables categóricas se transformaron en variables indicadoras para que el conjunto
        de datos pudiera utilizarse en el modelado estadístico.
        """)
else:
    st.warning("No se detectaron las variables categóricas configuradas en el dataset.")

# -----------------------------
# Partición train / test
# -----------------------------
st.markdown("## Separación en entrenamiento y prueba")

left, right = st.columns([1, 1.2])

with left:
    st.markdown("""
Se separó el conjunto de datos en dos subconjuntos:

- **entrenamiento**, para ajustar el modelo;
- **prueba**, para evaluar su desempeño en datos no vistos.

La partición se realizó mediante **muestreo estratificado**, con el fin de preservar
la proporción de clases en ambos subconjuntos.
""")

with right:
    st.markdown("### Tamaño de los subconjuntos")
    st.dataframe(size_df, use_container_width=True, hide_index=True)

# -----------------------------
# Desbalance y balanceo
# -----------------------------
st.markdown("## Desbalance de clases y balanceo del entrenamiento")

st.markdown("""
La variable objetivo presenta una distribución desigual entre clases. Esta situación es común
en problemas de abandono de clientes y puede dificultar el aprendizaje del modelo.

Para enfrentar este problema, se aplicó un **submuestreo de la clase mayoritaria**
**solo dentro del conjunto de entrenamiento**.
""")

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("### Distribuciones por subconjunto")
    st.dataframe(
        dist_df[["subset", TARGET_COL, "frecuencia", "porcentaje"]],
        use_container_width=True,
        hide_index=True
    )

with c2:
    st.markdown("### Comparación visual")
    fig = plot_class_distributions(dist_df, TARGET_COL)
    st.pyplot(fig)

st.warning("""
El conjunto de prueba **no** se balanceó. Esto permite evaluar el modelo sobre una distribución
de clases más cercana a la que aparece en el problema real.
""")

# -----------------------------
# Resultado del preproceso
# -----------------------------
st.markdown("## Resultado del preproceso")

r1, r2, r3 = st.columns(3)

r1.metric("Variables tras recodificación", f"{results['X_encoded'].shape[1]:,}")
r2.metric("Registros en train original", f"{len(results['y_train']):,}")
r3.metric("Registros en train balanceado", f"{len(results['y_train_bal']):,}")

st.markdown(
    """
    <div style="
        background-color:#f9fafc;
        padding:1rem 1.2rem;
        border-radius:14px;
        border:1px solid #dfe6f2;
        margin-top:0.5rem;
    ">
        <p style="margin-bottom:0;">
        <b>Flujo seguido:</b> base analítica → recodificación de variables categóricas →
        partición estratificada en entrenamiento y prueba → balanceo del conjunto de entrenamiento →
        datos listos para modelado.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.success("""
Con esta preparación, el conjunto de entrenamiento queda listo para ajustar el modelo,
mientras que el conjunto de prueba permanece reservado para una evaluación más confiable.
""")