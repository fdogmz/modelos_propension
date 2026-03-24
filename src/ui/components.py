import streamlit as st


def page_header(title: str, description: str | None = None) -> None:
    """Renderiza un encabezado consistente para cada pagina."""
    st.title(title)
    if description:
        st.caption(description)


def info_box(message: str, kind: str = "info") -> None:
    """Renderiza cajas informativas simples segun el tipo."""
    if kind == "success":
        st.success(message)
    elif kind == "warning":
        st.warning(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)


def placeholder_block(title: str, description: str = "Contenido en construccion.") -> None:
    """Bloque temporal para secciones aun no implementadas."""
    st.subheader(title)
    st.write(description)
    st.divider()
