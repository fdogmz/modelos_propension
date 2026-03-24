import streamlit as st
from src.ui.components import page_header

page_header(
    "1. Introducción",
    "Priorizar clientes cuando no se puede actuar sobre todos"
)

st.markdown("## Tres situaciones frecuentes en marketing")

st.markdown("""
**¿Qué hacer cuando el presupuesto no alcanza para actuar sobre todos los clientes?**

En muchos contextos de marketing, la organización necesita decidir **sobre quién conviene actuar primero**.
Las siguientes situaciones ilustran una misma necesidad analítica: **priorizar clientes antes de intervenir**.
""")

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
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        ">
            <h4 style="margin-top:0;">🛒 ¿Quiénes podrían comprar?</h4>
            <p>
            Una empresa quiere promocionar un nuevo producto, pero contactar a toda la base de clientes
            implicaría un costo elevado en tiempo, personal y presupuesto comercial.
            </p>
            <p>
            Necesita identificar a quién conviene dirigir primero la oferta para aumentar la probabilidad
            de conversión.
            </p>
            <p><b>Pregunta de negocio:</b><br>
            ¿Qué clientes tienen mayor probabilidad de comprar?</p>
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
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        ">
            <h4 style="margin-top:0;">⚠️ ¿Quiénes podrían abandonar?</h4>
            <p>
            Una empresa observa que algunos clientes dejan de renovar el servicio, pero no puede ofrecer
            incentivos de retención a toda la cartera sin elevar demasiado el costo.
            </p>
            <p>
            Necesita anticipar qué clientes presentan mayor riesgo de abandono para intervenir de forma selectiva.
            </p>
            <p><b>Pregunta de negocio:</b><br>
            ¿Qué clientes tienen mayor probabilidad de abandonar?</p>
        </div>
        """,
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
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        ">
            <h4 style="margin-top:0;">📩 ¿Quiénes podrían responder?</h4>
            <p>
            Una organización lanzará una campaña promocional, pero sabe que solo una fracción de los clientes
            responderá favorablemente.
            </p>
            <p>
            Para no diluir el presupuesto, necesita enfocar la campaña en quienes tienen mayor probabilidad
            de reaccionar positivamente.
            </p>
            <p><b>Pregunta de negocio:</b><br>
            ¿Qué clientes tienen mayor probabilidad de responder a la campaña?</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("## ¿Qué tienen en común estas decisiones?")

st.markdown("""
En los tres casos aparece la misma lógica de negocio:

- existe una **acción de marketing costosa**;
- el presupuesto disponible es **limitado**;
- no conviene actuar sobre toda la base de clientes;
- se requiere **priorizar** antes de intervenir.

Por ello, la pregunta central no es solo quién pertenece a una categoría, sino
**qué clientes muestran mayor probabilidad de generar el comportamiento de interés**.
""")

st.success("""
A este tipo de enfoque se le conoce como **modelo de propensión**:
una herramienta que estima la probabilidad de que un cliente compre, abandone o responda,
con el fin de orientar recursos limitados hacia **acciones más precisas y focalizadas**.
""")

with st.expander("¿Qué verás en las siguientes páginas?"):
    st.markdown(
        """
        - **Fundamentos:** qué es un score de propensión y cómo debe interpretarse.
        - **Desbalance y preparación:** por qué la clase positiva suele ser minoritaria.
        - **Modelado y evaluación:** métricas estadísticas para evaluar desempeño.
        - **Lift, ganancia y decisión:** cómo conectar el modelo con decisiones de negocio.
        - **Ejemplo interactivo:** una demostración práctica.
        - **Aplicaciones:** contextos de uso en marketing analítico.
        """
    )