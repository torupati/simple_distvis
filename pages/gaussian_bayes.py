import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

st.title("Gaussian Prior and Bayesian Updating")

# 1. slider for parameters
st.header("Set Prior and Likelihood Parameters")
mu_prior = st.slider("Prior Mean (μ_prior)", -10.0, 10.0, 0.0, step=0.1)
sigma_prior = st.slider("Prior Standard Deviation (σ_prior)", 0.1, 10.0, 1.0, step=0.1)

st.header("Set Likelihood Parameters")
mu_likelihood = st.slider("Likelihood Mean (μ_likelihood)", -10.0, 10.0, 2.0, step=0.1)
sigma_likelihood = st.slider("Likelihood Standard Deviation (σ_likelihood)", 0.1, 10.0, 1.0, step=0.1)

# 2. compute posterior parameters
def compute_posterior(mu_prior, sigma_prior, mu_likelihood, sigma_likelihood):
    var_prior = sigma_prior ** 2
    var_likelihood = sigma_likelihood ** 2
    
    var_posterior = 1 / (1/var_prior + 1/var_likelihood)
    mu_posterior = var_posterior * (mu_prior/var_prior + mu_likelihood/var_likelihood)
    
    sigma_posterior = np.sqrt(var_posterior)
    
    return mu_posterior, sigma_posterior

mu_posterior, sigma_posterior = compute_posterior(mu_prior, sigma_prior, mu_likelihood, sigma_likelihood)

table_data = {
    "Parameter": ["Prior Mean (μ_prior)", "Prior Std Dev (σ_prior)",
                  "Likelihood Mean (μ_likelihood)", "Likelihood Std Dev (σ_likelihood)",
                  "Posterior Mean (μ_posterior)", "Posterior Std Dev (σ_posterior)"],
    "Value": [mu_prior, sigma_prior, mu_likelihood, sigma_likelihood, mu_posterior, sigma_posterior]
}
df = pd.DataFrame(table_data)
st.table(df)

# 3. plot prior, likelihood, posterior
st.header("Distributions")
x = np.linspace(-10, 10, 400)
prior_pdf = norm.pdf(x, mu_prior, sigma_prior)
likelihood_pdf = norm.pdf(x, mu_likelihood, sigma_likelihood)
posterior_pdf = norm.pdf(x, mu_posterior, sigma_posterior)
fig, ax = plt.subplots()
ax.plot(x, prior_pdf, label="Prior", color='blue')
ax.plot(x, likelihood_pdf, label="Likelihood", color='green')
ax.plot(x, posterior_pdf, label="Posterior", color='red')
ax.fill_between(x, prior_pdf, color='blue', alpha=0.1)
ax.fill_between(x, likelihood_pdf, color='green', alpha=0.1)
ax.fill_between(x, posterior_pdf, color='red', alpha=0.1)
ax.set_title("Prior, Likelihood, and Posterior Distributions")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)
