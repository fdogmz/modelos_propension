import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.ui.components import page_header

# -----------------------------
# Configuración
# -----------------------------
DATA_PATH = Path("data/processed/churn_case_analysis_base.parquet")
TARGET_COL = "CHURN"   # Cambia a "CHURN" si ése es el nombre real en tu dataset

st.set_page_config(layout="wide")

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def classify_columns(df: pd.DataFrame, target_col: str):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    return numeric_cols, categorical_cols

def plot_target_distribution(df: pd.DataFrame, target_col: str):
    counts = df[target_col].value_counts(dropna=False).sort_index()

    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Distribución de la variable objetivo")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Frecuencia")
    return fig

# -----------------------------
# Carga
# -----------------------------
page_header(
    "4. Exploración del conjunto de datos",
    "Estructura, completitud y distribución de la variable objetivo"
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

numeric_cols, categorical_cols = classify_columns(df, TARGET_COL)

# -----------------------------
# Introducción
# -----------------------------
st.markdown("""
En esta sección se revisa la **base analítica** del caso de estudio. El objetivo es conocer
su estructura general, las variables disponibles, la completitud de los datos y la distribución
de la variable objetivo antes de pasar a etapas posteriores de preparación y modelado.
""")

# -----------------------------
# Vista previa
# -----------------------------
st.markdown("## Primeras observaciones")

st.markdown("""
La siguiente tabla muestra una vista previa del conjunto de datos ya filtrado para el análisis.
Cada fila representa un cliente y cada columna una característica histórica o la variable objetivo.
""")

st.dataframe(df.head(10), use_container_width=True)

# -----------------------------
# Métricas resumen
# -----------------------------
st.markdown("## Vista general del conjunto de datos")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Registros", f"{df.shape[0]:,}")
m2.metric("Columnas", f"{df.shape[1]:,}")
m3.metric("Variables numéricas", f"{len(numeric_cols):,}")
m4.metric("Variables categóricas", f"{len(categorical_cols):,}")


# -----------------------------
# Columnas seleccionadas
# -----------------------------
st.markdown("## Variables disponibles")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div style="
            background-color:#ffffff;
            padding:1rem;
            border-radius:14px;
            border:1px solid #e6eaf2;
            height:100%;
        ">
            <h4 style="margin-top:0;">Variable objetivo</h4>
            <p style="margin-bottom:0;"><b>{}</b></p>
        </div>
        """.format(TARGET_COL),
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div style="
            background-color:#ffffff;
            padding:1rem;
            border-radius:14px;
            border:1px solid #e6eaf2;
            height:100%;
        ">
            <h4 style="margin-top:0;">Variables numéricas</h4>
            <p style="margin-bottom:0;">{}</p>
        </div>
        """.format(", ".join(numeric_cols) if numeric_cols else "No hay"),
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div style="
            background-color:#ffffff;
            padding:1rem;
            border-radius:14px;
            border:1px solid #e6eaf2;
            height:100%;
        ">
            <h4 style="margin-top:0;">Variables categóricas</h4>
            <p style="margin-bottom:0;">{}</p>
        </div>
        """.format(", ".join(categorical_cols) if categorical_cols else "No hay"),
        unsafe_allow_html=True
    )

with st.expander("Ver diccionario de datos de las variables seleccionadas"):
    predictors_dict = pd.DataFrame(
        [
            ("GENDER", "Género del cliente"),
            ("MARITAL_STATUS", "Estado civil del cliente"),
            ("AGE", "Edad del cliente"),
            (
                "NUM_CARDS_Sum",
                "Número total de tarjetas adicionales asociadas a todas las tarjetas principales que el cliente tuvo durante el periodo de observación",
            ),
            (
                "TENURE_CUSTOMER",
                "Meses transcurridos desde el primer registro del cliente",
            ),
            (
                "AVG_SPENDING_AMOUNT",
                "Gasto promedio mensual (compras y disposiciones en efectivo) durante los 12 meses del periodo de observación, considerando todas las tarjetas que el cliente tuvo en dicho periodo",
            ),
            (
                "SPENDING_RECENCY",
                "Meses transcurridos desde la última transacción de gasto",
            ),
            (
                "SPENDING_FREQUENCY",
                "Porcentaje de meses con transacciones de gasto",
            ),
            (
                "DELTA_SPENDING_FREQUENCY",
                "Cambio relativo (incremento o decremento porcentual) en la frecuencia de gasto durante los últimos 4 meses del periodo de observación",
            ),
            (
                "DELTA_SPENDING_AMOUNT",
                "Cambio relativo (incremento o decremento porcentual) en el gasto promedio mensual durante los últimos 4 meses del periodo de observación",
            ),
            (
                "AVG_TRX",
                "Número promedio mensual de transacciones de gasto durante los 12 meses del periodo de observación",
            ),
            (
                "DELTA_TRX",
                "Cambio relativo (incremento o decremento porcentual) en el número promedio mensual de transacciones durante los últimos 4 meses del periodo de observación",
            ),
            (
                "AVG_TRX_AMOUNT",
                "Monto promedio gastado por transacción",
            ),
            (
                "AVG_BALANCES",
                "Saldo promedio mensual durante los 12 meses del periodo de observación",
            ),
            (
                "BALANCES_FREQUENCY",
                "Porcentaje de meses con saldo",
            ),
            (
                "SPENDING_LIMIT_RATIO",
                "Razón entre el monto gastado y el límite de crédito",
            ),
            (
                "LIMIT_USAGE",
                "Razón entre los saldos y el límite de crédito",
            ),
            (
                "END_START_CARDS_DELTAS",
                "Diferencia en el número de tarjetas abiertas entre el final y el inicio del periodo de observación",
            ),
            (
                "END_WITHIN_CARDS_DELTAS",
                "Cambio en el número de tarjetas abiertas al final del periodo de observación respecto al total de tarjetas que el cliente tuvo en algún momento dentro del mismo periodo",
            ),
        ],
        columns=["Variable", "Descripción"],
    )

    st.markdown("### Predictores")
    st.dataframe(predictors_dict, use_container_width=True, hide_index=True)

    st.markdown("### Variable objetivo")
    target_dict = pd.DataFrame(
        [
            (
                "CHURN",
                "Indica si el cliente abandonó la institución en la ventana futura de observación",
            )
        ],
        columns=["Variable", "Descripción"],
    )
    st.dataframe(target_dict, use_container_width=True, hide_index=True)


# -----------------------------
# Calidad básica: faltantes y tipos
# -----------------------------
st.markdown("## Calidad básica de los datos")

left, right = st.columns([1, 1.2])

with left:
    missing_df = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "variable", 0: "faltantes"})
    )
    missing_df["porcentaje_faltantes"] = 100 * missing_df["faltantes"] / len(df)
    missing_df = missing_df.sort_values(
        by=["faltantes", "variable"], ascending=[False, True]
    )

    st.markdown("### Valores faltantes por variable")
    st.dataframe(missing_df, use_container_width=True)

with right:
    dtypes_df = pd.DataFrame({
        "variable": df.columns,
        "tipo": df.dtypes.astype(str).values
    })

    st.markdown("### Tipos de datos")
    st.dataframe(dtypes_df, use_container_width=True)

if missing_df["faltantes"].sum() == 0:
    st.success("No se detectaron valores faltantes en esta base analítica.")
else:
    st.warning("Existen variables con valores faltantes. Conviene revisarlas antes del modelado.")

# -----------------------------
# Distribución de la variable objetivo
# -----------------------------
st.markdown("## Distribución de la variable objetivo")

target_counts = (
    df[TARGET_COL]
    .value_counts(dropna=False)
    .rename_axis(TARGET_COL)
    .reset_index(name="frecuencia")
    .sort_values(by=TARGET_COL)
)

target_counts["porcentaje"] = 100 * target_counts["frecuencia"] / len(df)

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("### Conteos y proporciones")
    st.dataframe(target_counts, use_container_width=True)

with c2:
    st.markdown("### Visualización")
    fig = plot_target_distribution(df, TARGET_COL)
    st.pyplot(fig)

st.info("""
La distribución de la variable objetivo permite identificar si el problema presenta o no
**desbalance de clases**, una característica muy común en modelos de propensión.
""")

# -----------------------------
# Resumen descriptivo
# -----------------------------
st.markdown("## Resumen descriptivo de variables numéricas")

if numeric_cols:
    selected_num_cols = st.multiselect(
        "Selecciona variables numéricas para resumir:",
        options=numeric_cols,
        default=numeric_cols[: min(6, len(numeric_cols))]
    )

    if selected_num_cols:
        desc = df[selected_num_cols].describe().T
        desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        st.dataframe(desc, use_container_width=True)
    else:
        st.info("Selecciona al menos una variable numérica para mostrar su resumen descriptivo.")
else:
    st.info("No se detectaron variables numéricas para resumir.")

# -----------------------------
# Cierre
# -----------------------------
st.success("""
Esta exploración inicial permite verificar que la base analítica ya está organizada para el caso
de estudio. En la siguiente etapa puede revisarse con mayor detalle la preparación de los datos
para modelado, incluyendo la partición en subconjuntos y el tratamiento del desbalance de clases.
""")