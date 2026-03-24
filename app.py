import streamlit as st

from src.ui.components import info_box, page_header
from src.ui.styles import apply_base_styles


st.set_page_config(page_title="Modelos de propensión", layout="wide")


def main() -> None:
    apply_base_styles()

    page_header(
        "Modelos de propensión",
        "Aplicación educativa para explorar, de extremo a extremo, el desarrollo y uso de modelos de propensión en analítica del marketing.",
    )

    st.markdown(
        """
Esta aplicación presenta un recorrido completo que va desde la motivación de los modelos de propensión hasta su interpretación en términos de valor de negocio.

A lo largo de la app se revisa cómo:

- un problema de marketing con presupuesto limitado puede traducirse a un problema de modelado;
- se construye un conjunto de datos a partir de información histórica y una variable objetivo futura;
- se prepara la base analítica para modelado;
- se ajustan y evalúan distintos modelos predictivos;
- y se utilizan herramientas como el **lift** y la **ganancia acumulada** para apoyar decisiones de negocio.
        """
    )

    info_box(
        "Usa el menú lateral para navegar entre las páginas de contenido.",
        kind="info",
    )

    st.subheader("Estructura de la app")
    st.markdown(
        """
1. Introducción  
2. Fundamentos de los modelos de propensión  
3. Caso de estudio: abandono de clientes  
4. Exploración del conjunto de datos  
5. Preparación de los datos para modelado  
6. Modelado con regresión logística  
7. Modelado con Random Forest  
8. Comparación de modelos y valor de negocio  
9. Conclusiones y cierre
        """
    )


if __name__ == "__main__":
    main()