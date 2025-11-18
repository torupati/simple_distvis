import logging

import streamlit as st

logging.getLogger("watchdog").setLevel(logging.WARNING)

st.set_page_config(page_title="App", page_icon="🌊")

st.title("Welcome to Simple Probabilistic Distribution Visualizer 🌊")
st.write(
    """
    Use the sidebar to select a page.
    - **Gaussian 1D Posterior**: Explore Bayesian updating with Gaussian priors and likelihoods.
    - **Beta Bayes**: Explore Beta-Binomial conjugate prior analysis.
    - **Diffusion Process**: Visualize diffusion processes with animated Gaussian distributions.
    - **Normal Distribution 1D**: Interactive exploration of normal distributions.
    - **Square Error Decomposition**: Analyze bias-variance tradeoff in 1D and 2D.
    """
)
