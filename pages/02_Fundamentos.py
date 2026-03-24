import streamlit as st
import pandas as pd
from src.ui.components import page_header

page_header(
    "2. Fundamentos de los modelos de propensión",
    "Del problema de negocio a la construcción del conjunto de datos"
)

st.markdown("## ¿Qué es un modelo de propensión?")

st.markdown("""
Un **modelo de propensión** es una herramienta analítica que estima la probabilidad de que un cliente
realice un comportamiento de interés en un periodo futuro, como abandonar la relación comercial
o responder favorablemente a una campaña.
""")

st.info("""
En términos prácticos, estos modelos ayudan a **priorizar clientes** cuando el presupuesto no permite
actuar sobre toda la base al mismo tiempo.
""")

st.markdown("## ¿Qué responde un modelo de propensión?")

st.markdown("""
Este tipo de modelo busca responder una pregunta como la siguiente:

**¿Qué tan probable es que un cliente realice una acción de interés en un periodo futuro?**
""")

st.markdown("""
Para responderla, el problema de negocio debe traducirse a un conjunto de datos estructurado:
se observan características históricas del cliente y después se registra si el evento ocurre o no
en una ventana futura.
""")

st.markdown(
    """
    <div style="
        background-color:#f5f7fb;
        padding:1rem 1.2rem;
        border-radius:14px;
        border:1px solid #d9e2f1;
        margin: 1rem 0 1rem 0;
    ">
        <h4 style="margin-top:0;">Dos aplicaciones frecuentes</h4>
        <p style="margin-bottom:0;">
        A continuación se muestran dos ejemplos de cómo un problema de propensión
        puede convertirse en un conjunto de datos listo para modelarse.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs([
    "⚠️ Abandono de clientes",
    "📩 Respuesta a campaña"
])

with tab1:
    st.markdown(
        """
        <div style="
            background-color:#ffffff;
            padding:1rem 1.2rem;
            border-radius:14px;
            border-left:6px solid #ff8a65;
            border:1px solid #e6eaf2;
            margin-bottom:1rem;
        ">
            <h3 style="margin-top:0;">Abandono de clientes</h3>
            <p style="margin-bottom:0;">
            <b>Objetivo de negocio:</b> identificar qué clientes presentan mayor riesgo de abandonar
            para enfocar acciones de retención de forma selectiva.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

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
                <h4 style="margin-top:0;">Cómo se construyen los datos</h4>
                <p>
                Se parte de un subconjunto de clientes que se encuentran activos al inicio del análisis.
                Durante una <b>ventana histórica</b> de observación, por ejemplo de 6 o 12 meses,
                se resumen indicadores de comportamiento como actividad reciente, compras, frecuencia,
                gasto y señales de menor involucramiento.
                </p>
                <p style="margin-bottom:0;">
                Después, en una <b>ventana futura</b>, se registra si el cliente continúa activo
                o si abandona.
                </p>
            </div>
            """,
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
                <h4 style="margin-top:0;">Ejemplo de variables</h4>
                <p>
                - recencia de última compra<br>
                - frecuencia de compra en 12 meses<br>
                - gasto acumulado en 12 meses<br>
                - ticket promedio<br>
                - meses sin actividad<br>
                - tendencia del gasto
                </p>
                <h4>Variable objetivo</h4>
                <p style="margin-bottom:0;">
                <b>1</b>: el cliente abandona en la ventana futura<br>
                <b>0</b>: el cliente permanece activo en la ventana futura
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Ejemplo sintético del conjunto de datos")

    df_abandono = pd.DataFrame({
        "cliente_id": [201, 202, 203, 204, 205],
        "recencia_dias": [95, 18, 120, 40, 10],
        "frecuencia_12m": [2, 11, 1, 7, 13],
        "gasto_12m": [1500, 9200, 600, 5400, 11000],
        "ticket_promedio": [750, 836, 600, 771, 846],
        "meses_sin_compra": [4, 0, 5, 1, 0],
        "tendencia_gasto": [-0.45, 0.10, -0.60, -0.15, 0.12],
        "abandono": [1, 0, 1, 0, 0],
    })

    st.table(df_abandono)

    st.caption("""
    Cada fila representa a un cliente activo al inicio del análisis. Las columnas resumen
    su comportamiento histórico, mientras que `abandono` indica si el cliente deja de estar activo
    en la ventana posterior de observación.
    """)

with tab2:
    st.markdown(
        """
        <div style="
            background-color:#ffffff;
            padding:1rem 1.2rem;
            border-radius:14px;
            border-left:6px solid #42a5f5;
            border:1px solid #e6eaf2;
            margin-bottom:1rem;
        ">
            <h3 style="margin-top:0;">Respuesta a campaña</h3>
            <p style="margin-bottom:0;">
            <b>Objetivo de negocio:</b> identificar qué clientes tienen mayor probabilidad de responder
            favorablemente a una campaña para concentrar el presupuesto promocional en los segmentos
            con mayor potencial.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

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
                <h4 style="margin-top:0;">Cómo se construyen los datos</h4>
                <p>
                Los datos pueden provenir de campañas previas o de un <b>piloto de campaña</b>
                realizado sobre un subconjunto de clientes.
                </p>
                <p>
                Por ejemplo, la empresa puede seleccionar aleatoriamente un grupo de clientes,
                aplicar la campaña y registrar quién responde y quién no.
                </p>
                <p style="margin-bottom:0;">
                A partir de las características históricas de esos clientes y del resultado observado,
                se construye un conjunto de datos para estimar la probabilidad de respuesta
                en campañas futuras.
                </p>
            </div>
            """,
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
                <h4 style="margin-top:0;">Ejemplo de variables</h4>
                <p>
                - aperturas previas de correo<br>
                - clics en campañas anteriores<br>
                - compras recientes<br>
                - gasto acumulado<br>
                - uso de promociones<br>
                - canal preferido
                </p>
                <h4>Variable objetivo</h4>
                <p style="margin-bottom:0;">
                <b>1</b>: el cliente respondió a la campaña<br>
                <b>0</b>: el cliente no respondió a la campaña
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Ejemplo sintético del conjunto de datos")

    df_campana = pd.DataFrame({
        "cliente_id": [301, 302, 303, 304, 305],
        "aperturas_previas": [8, 1, 5, 0, 6],
        "clics_previos": [3, 0, 2, 0, 4],
        "compras_6m": [5, 1, 4, 0, 6],
        "gasto_6m": [4200, 900, 3100, 0, 5800],
        "uso_promociones": [1, 0, 1, 0, 1],
        "canal_preferido": ["email", "sms", "email", "sms", "email"],
        "respuesta_campana": [1, 0, 1, 0, 1],
    })

    st.table(df_campana)

    st.caption("""
    En este ejemplo, `respuesta_campana` registra si el cliente respondió o no a una campaña observada
    previamente. Esa información, combinada con variables históricas, permite entrenar un modelo
    para priorizar clientes en campañas posteriores.
    """)

st.markdown("## ¿Qué comparten ambos casos?")

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
        En ambos casos, cada fila representa un cliente, las columnas describen su comportamiento histórico
        y la variable objetivo indica si ocurre o no un evento futuro relevante para el negocio.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.success("""
A partir de esta estructura, los modelos de propensión aprenden patrones en los datos históricos
para estimar probabilidades útiles en decisiones de retención y campañas.
""")