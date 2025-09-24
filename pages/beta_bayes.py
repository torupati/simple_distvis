"""
A Streamlit app to visualize Bayesian updating with a Beta prior and Bernoulli likelihood.
Users can adjust the parameters of the Beta prior and the number of successes/failures in the Bernoulli trials.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, bernoulli

st.title("Beta-Bernoulli Bayesian Updating")

st.header("Set Prior Beta Parameters")
alpha_prior = st.slider(r"Prior $\alpha$ (successes + 1)", 0.1, 10.0, 2.0, step=0.1)
beta_prior = st.slider(r"Prior $\beta$ (failures + 1)", 0.1, 10.0, 2.0, step=0.1)

st.header("Set Bernoulli Likelihood Parameters")
n_trials = st.slider("Number of Bernoulli trials", 1, 100, 10, step=1)
n_success = st.slider("Number of successes", 0, n_trials, 5, step=1)

# Posterior parameters
alpha_post = alpha_prior + n_success
beta_post = beta_prior + (n_trials - n_success)

# Plotting
x = np.linspace(0, 1, 400)
prior_pdf = beta.pdf(x, alpha_prior, beta_prior)
likelihood_pdf = x**n_success * (1-x)**(n_trials-n_success)
likelihood_pdf /= np.max(likelihood_pdf)  # normalize for display
posterior_pdf = beta.pdf(x, alpha_post, beta_post)

st.header("Show/Hide Distributions")
show_prior = st.checkbox("Show Prior", value=True)
show_likelihood = st.checkbox("Show Likelihood", value=True)
show_posterior = st.checkbox("Show Posterior", value=True)

fig, ax = plt.subplots(figsize=(8, 5))
if show_prior:
    ax.plot(x, prior_pdf, label="Prior (Beta)", color='blue')
    ax.fill_between(x, prior_pdf, color='blue', alpha=0.2)
if show_likelihood:
    ax.plot(x, likelihood_pdf, label="Likelihood (Bernoulli)", color='green')
    ax.fill_between(x, likelihood_pdf, color='green', alpha=0.2)
if show_posterior:
    ax.plot(x, posterior_pdf, label="Posterior (Beta)", color='red')
    ax.fill_between(x, posterior_pdf, color='red', alpha=0.2)
ax.set_xlabel("Parameter θ")
ax.set_ylabel("Density")
ax.set_title("Beta-Bernoulli Bayesian Update")
ax.legend()
ax.grid(True)
st.pyplot(fig)