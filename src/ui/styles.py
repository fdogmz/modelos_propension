import streamlit as st


def apply_base_styles() -> None:
    """Aplica estilos minimos para mejorar legibilidad sin complejidad visual."""
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
