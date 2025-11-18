import logging

import streamlit as st

logging.getLogger("watchdog").setLevel(logging.WARNING)

st.set_page_config(page_title="App", page_icon="🌊")

st.title("Welcome to App")
st.write(
    """
    Use the sidebar to select a page.
    - **Gaussian 1D Posterior**: Explore Bayesian updating with Gaussian priors and likelihoods.
    - **Gaussian 2D Posterior**: Visualize Bayesian updating in two dimensions.
    - **Bernoulli-Beta Posterior**: Understand Bayesian updating with Bernoulli trials and Beta priors.
    - **Poisson-Gamma Posterior**: Analyze Bayesian updating with Poisson data and Gamma priors.
    - **Conjugate Priors**: Overview of conjugate prior relationships.
    - **Bias and Variance**: bias-variance tradeoff visualization.
    """
)
