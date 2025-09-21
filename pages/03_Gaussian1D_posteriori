from logging import getLogger, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

st.title("Gaussian Prior and Bayesian Updating")

# 1. slider for parameters
st.header(r"Prior $P_{\theta_0}(x) = \mathcal{N}(x | \mu_0, \sigma^2)$ Parameters")
mu_prior = st.slider(r"Prior position $\mu_0$", -10.0, 10.0, 0.0, step=0.1)
sigma_prior = st.slider(r"Prior position confidence $\sigma_0$", 0.1, 10.0, 1.0, step=0.1)

st.header(r"Observation $y$ and Observation Probability $P(y|x)=\mathcal{N}(y|x, \nu^2)$ (Likelihood $L_y(x)$)")
mu_likelihood = st.slider(r"Observation $Y=y$", -10.0, 10.0, 2.0, step=0.1)
sigma_likelihood = st.slider("Likelihood Standard Deviation (σ_likelihood)", 0.1, 10.0, 1.0, step=0.1)

# 2. compute posterior parameters
def compute_posterior(mu_0, sigma_0, mu_likelihood, sigma_likelihood):
    """Calculate posterior parameters for 1D Gaussian prior and likelihood.

    Args:
        mu_0 (float): mean of the prior Gaussian
        sigma_0 (float): standard deviation of the prior Gaussian
        mu_likelihood (_type_): _description_
        sigma_likelihood (_type_): _description_

    Returns:
        mu_posterior (float): mean of the posterior Gaussian
        sigma_posterior (float): standard deviation of the posterior Gaussian
    """
    var_likelihood = sigma_likelihood ** 2
    
    var_posterior = 1 / (1.0/sigma_0**2 + 1/var_likelihood)
    mu_posterior = var_posterior * (mu_0/sigma_0**2 + mu_likelihood/var_likelihood)
    
    sigma_posterior = np.sqrt(var_posterior)
    
    return mu_posterior, sigma_posterior

mu_posterior, sigma_posterior = compute_posterior(mu_prior, sigma_prior, mu_likelihood, sigma_likelihood)

table_data = {
    "Parameter": [
        "prioir position ",
        r"prior confidence ($σ_0$ is std dev)",
        "observation (y)",
        "Likelihood Std Dev (σ_likelihood)",
        "Posterior position (μ_posterior)",
        r"Posterior confidence ($\sigma_1$ is std dev)"
    ],
    "Value": [
        mu_prior, sigma_prior, mu_likelihood, sigma_likelihood, mu_posterior, sigma_posterior
    ],
    "Formula": [
        r"$\mu_0$",
        r"$\sigma_0$",
        r"$y$",
        r"$\nu$",
        r"$\mu_1$",
        r"$\sigma_1$"
    ]
}
df = pd.DataFrame(table_data)
st.table(df)

# 3. plot prior, likelihood, posterior
st.header("Distributions")
x_min, x_max = -10, 10
x = np.linspace(x_min, x_max, 400)
prior_pdf = norm.pdf(x, mu_prior, sigma_prior)
likelihood_pdf = norm.pdf(x, mu_likelihood, sigma_likelihood)
joint_pdf = prior_pdf * likelihood_pdf
posterior_pdf = norm.pdf(x, mu_posterior, sigma_posterior)
# check boxes to show/hide distributions
show_prior = st.checkbox("Prior $P_{\\theta_0}(x)$", value=True)
show_likelihood = st.checkbox("Likelihood $L_y(x)=P(y|x)$", value=False)
show_joint = st.checkbox("Joint $P_\\theta(x)P(y|x)$", value=False)
show_posterior = st.checkbox("Posterior $P_{\\theta_1}(x|y)$", value=True)
show_grid = st.checkbox("Show Grid", value=True)

fig, ax = plt.subplots(figsize=(16, 6))
if show_prior:
    ax.plot(x, prior_pdf, label="Prior", color='blue')
    ax.fill_between(x, prior_pdf, color='blue', alpha=0.1)
if show_likelihood:
    ax.plot(x, likelihood_pdf, label="Likelihood", color='green')
    ax.fill_between(x, likelihood_pdf, color='green', alpha=0.1)
if show_joint:
    ax.plot(x, joint_pdf, label="Joint Probability", color='orange')
    ax.fill_between(x, joint_pdf, color='orange', alpha=0.1)
if show_posterior:
    ax.plot(x, posterior_pdf, label="Posterior", color='red')
    ax.fill_between(x, posterior_pdf, color='red', alpha=0.1)
if show_grid:
    ax.grid(True)
ax.set_title("Prior, Likelihood, and Posterior Distributions")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_xlim(x_min, x_max)
ax.legend()
st.pyplot(fig)

st.markdown(r"""
**Posterior formulas:**

$$
\mu_1 = \frac{\mu_0/\sigma_0^2 + \mu_y/\sigma_y^2}{1/\sigma_0^2 + 1/\sigma_y^2}
$$

$$
\sigma_1 = \sqrt{\frac{1}{1/\sigma_0^2 + 1/\sigma_y^2}}
$$
""")
