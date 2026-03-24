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
    with st.container(border=True):
        st.markdown("#### 📩 ¿Quiénes podrían responder?")
        st.markdown("Una organización lanzará una campaña promocional, pero sabe que solo una fracción de los clientes responderá favorablemente.")
        st.markdown("Para no diluir el presupuesto, necesita enfocar la campaña en quienes tienen mayor probabilidad de reaccionar positivamente.")
        st.markdown("**Pregunta de negocio:**\n\n¿Qué clientes tienen mayor probabilidad de responder a la campaña?")

with col2:
    with st.container(border=True):
        st.markdown("#### ⚠️ ¿Quiénes podrían abandonar?")
        st.markdown("Una empresa observa que algunos clientes dejan de renovar el servicio, pero no puede ofrecer incentivos de retención a toda la cartera sin elevar demasiado el costo.")
        st.markdown("Necesita anticipar qué clientes presentan mayor riesgo de abandono para intervenir de forma selectiva.")
        st.markdown("**Pregunta de negocio:**\n\n¿Qué clientes tienen mayor probabilidad de abandonar?")

with col3:
    with st.container(border=True):
        st.markdown("#### 🚀 ¿Quiénes harían un *Upgrade*?")
        st.markdown("Una empresa ofrece un servicio básico, pero contactar a todos los usuarios para ofrecerles el plan *Premium* agotaría el tiempo de los ejecutivos de cuenta.")
        st.markdown("Para maximizar los ingresos, necesita enfocar las llamadas comerciales en quienes están listos para adquirir un servicio más avanzado.")
        st.markdown("**Pregunta de negocio:**\n\n¿Qué clientes tienen mayor probabilidad de mejorar su plan de suscripción?")

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