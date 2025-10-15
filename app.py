import logging

import streamlit as st

logging.getLogger("watchdog").setLevel(logging.WARNING)

st.set_page_config(page_title="App", page_icon="🌊")

st.title("Welcome to App")
st.write(
    """
    Use the sidebar to select a page.
    - **Gaussian 1D Posterior**: Explore Bayesian updating with Gaussian priors and likelihoods.
    - **Gaussian Mixture Model (GMM)**: Explore GMMs and their applications.
    - **K-Means Clustering**: Perform K-Means clustering on your data and visualize the results.
    - **Bias and Variance**: bias-variance tradeoff visualization.
    """
)
