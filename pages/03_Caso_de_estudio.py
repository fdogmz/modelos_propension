import streamlit as st
from src.ui.components import page_header

page_header(
    "3. Caso de estudio: abandono de clientes",
    "Una aplicación de modelos de propensión en una institución bancaria"
)

st.markdown(
    """
    <div style="
        background-color:#f5f7fb;
        padding:1rem 1.2rem;
        border-radius:14px;
        border:1px solid #d9e2f1;
        margin-bottom:1rem;
    ">
        <h3 style="margin-top:0;">Del concepto al caso</h3>
        <p style="margin-bottom:0;">
        Después de revisar qué es un modelo de propensión y cómo se construyen sus datos,
        conviene analizar un caso concreto. En esta sección se presenta un estudio orientado
        a anticipar el abandono de clientes en una institución bancaria.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("## Contexto del caso")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown(
        """
        El caso corresponde a una **institución bancaria**. Los clientes analizados son
        **usuarios de tarjetas de crédito**, para quienes se dispone de información
        **demográfica** y de **uso**.

        El interés del estudio es anticipar qué clientes presentan mayor probabilidad
        de abandonar la institución, con el fin de apoyar acciones de retención
        más oportunas y focalizadas.
        """
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
            <h4 style="margin-top:0;">Unidad de análisis</h4>
            <p><b>Cliente bancario</b></p>
            <h4>Producto observado</h4>
            <p><b>Tarjeta de crédito</b></p>
            <h4>Tipo de variables</h4>
            <p style="margin-bottom:0;">
            Demográficas y de uso
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("## Objetivo de negocio")

st.info(
    "Identificar a los clientes con mayor propensión a abandonar la organización para "
    "priorizar acciones de retención, especialmente en los segmentos de mayor valor."
)

st.markdown("## ¿Qué se considera abandono?")

st.markdown(
    """
    <div style="
        background-color:#fff8f6;
        padding:1rem 1.2rem;
        border-radius:14px;
        border-left:6px solid #ff8a65;
        border:1px solid #f0d7d1;
        margin: 0.5rem 0 1rem 0;
    ">
        <h4 style="margin-top:0;">Definición operativa</h4>
        <p style="margin-bottom:0;">
        En este estudio, se considera como abandono a los clientes que
        <b>voluntaria y explícitamente cerraron sus cuentas</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("## Diseño temporal del estudio")

st.markdown(
    """
    Para construir el conjunto de datos se definieron dos etapas temporales:

    - una **ventana de observación de 12 meses**, en la que se registran las variables predictoras;
    - una **ventana posterior de 4 meses**, en la que se observa si el cliente abandona o permanece.
    """
)

t1, t2, t3 = st.columns([1.4, 0.35, 1.1])

with t1:
    st.markdown(
        """
        <div style="
            background-color:#ffffff;
            padding:1rem;
            border-radius:14px;
            border:1px solid #e6eaf2;
            text-align:center;
            min-height:130px;
        ">
            <h4 style="margin-top:0;">Ventana de observación</h4>
            <p style="font-size:1.1rem;"><b>12 meses</b></p>
            <p style="margin-bottom:0;">
            Se resumen variables históricas del cliente:
            actividad, uso, gasto, recencia, frecuencia y otras señales relevantes.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with t2:
    st.markdown(
        """
        <div style="
            text-align:center;
            font-size:2rem;
            padding-top:2.2rem;
            color:#5b657a;
        ">
            →
        </div>
        """,
        unsafe_allow_html=True
    )

with t3:
    st.markdown(
        """
        <div style="
            background-color:#ffffff;
            padding:1rem;
            border-radius:14px;
            border:1px solid #e6eaf2;
            text-align:center;
            min-height:130px;
        ">
            <h4 style="margin-top:0;">Ventana de resultado</h4>
            <p style="font-size:1.1rem;"><b>4 meses</b></p>
            <p style="margin-bottom:0;">
            Se registra si el cliente
            <b>permanece</b> o <b>abandona</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("## ¿Qué tipo de información se utiliza?")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div style="
            background-color:#ffffff;
            padding:1rem;
            border-radius:14px;
            border:1px solid #e6eaf2;
            height:100%;
        ">
            <h4 style="margin-top:0;">Variables demográficas</h4>
            <p style="margin-bottom:0;">
            Edad, género y estado civil, entre otras variables del cliente.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div style="
            background-color:#ffffff;
            padding:1rem;
            border-radius:14px;
            border:1px solid #e6eaf2;
            height:100%;
        ">
            <h4 style="margin-top:0;">Variables de comportamiento</h4>
            <p style="margin-bottom:0;">
            Recencia, frecuencia, gasto promedio y cambios recientes en el uso.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div style="
            background-color:#ffffff;
            padding:1rem;
            border-radius:14px;
            border:1px solid #e6eaf2;
            height:100%;
        ">
            <h4 style="margin-top:0;">Variables financieras</h4>
            <p style="margin-bottom:0;">
            Uso del límite de crédito, balances y otras razones financieras asociadas a la cuenta.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("## ¿Por qué este caso es útil para estudiar modelos de propensión?")

st.markdown(
    """
    Este caso ilustra con claridad la lógica general de un modelo de propensión:

    - cada fila representa a un cliente;
    - las variables describen su comportamiento histórico;
    - la variable objetivo refleja un evento futuro de interés;
    - y el resultado del modelo puede utilizarse para priorizar decisiones de negocio.
    """
)

st.success(
    "En las siguientes secciones se revisará con más detalle la estructura del conjunto de datos, "
    "las variables empleadas y la forma en que se construye y evalúa el modelo."
)