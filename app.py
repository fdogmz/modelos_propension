import streamlit as st

from src.ui.components import info_box, page_header
from src.ui.styles import apply_base_styles


st.set_page_config(page_title="Modelos de propension", layout="wide")


def main() -> None:
    apply_base_styles()

    page_header(
        "Modelos de propension",
        "Aplicacion educativa para explorar de extremo a extremo modelos de propension.",
    )

    st.markdown(
        """
Esta aplicacion presenta un flujo completo para:
- Entender fundamentos de clasificacion binaria.
- Tratar el desbalance de clases y preparar datos.
- Entrenar y evaluar modelos.
- Traducir resultados a decisiones de negocio.
        """
    )

    info_box(
        "Usa el menu lateral para navegar entre las 7 paginas de contenido.",
        kind="info",
    )

    st.subheader("Estructura de la app")
    st.markdown(
        """
1. Introduccion
2. Fundamentos
3. Desbalance y preparacion
4. Modelado y evaluacion
5. Lift, ganancia y decision
6. Ejemplo interactivo
7. Aplicaciones
        """
    )


if __name__ == "__main__":
    main()
