import streamlit as st
import logging

logging.getLogger("watchdog").setLevel(logging.WARNING)

st.set_page_config(page_title="Diffusion Animation App", page_icon="🌊")

st.title("Welcome to the Diffusion Animation App")
st.write(
    """
    Use the sidebar to select a page.
    - **Diffusion Animation**: See a 1D diffusion process animated.
    """
)