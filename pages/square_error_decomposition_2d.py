import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("2D Square Error Decomposition: Mean Square = (Mean)$^2$ + Variance")

# User input for mean and covariance
mean1 = st.slider("Mean (dimension 1)", -10.0, 10.0, 2.0, step=0.1)
mean2 = st.slider("Mean (dimension 2)", -10.0, 10.0, -1.0, step=0.1)
var1 = st.slider("Variance (dimension 1)", 0.1, 10.0, 2.0, step=0.1)
var2 = st.slider("Variance (dimension 2)", 0.1, 10.0, 1.0, step=0.1)
cov = st.slider("Covariance", -3.0, 3.0, 0.0, step=0.1)
n = 200

mean_vec = [mean1, mean2]
cov_mat = [[var1, cov], [cov, var2]]

# Generate random samples
np.random.seed(0)
x = np.random.multivariate_normal(mean_vec, cov_mat, n)

# Calculate statistics for each dimension
mean_x = np.mean(x, axis=0)
var_x = np.var(x, axis=0)
mse = np.mean(x**2, axis=0)
sum_formula = mean_x**2 + var_x
cov_sample = np.cov(x, rowvar=False)

# Display results
st.write(f"Sample Mean: {mean_x}")
st.write(f"Sample Variance: {var_x}")
st.write(f"Mean of $x^2$ (MSE): {mse}")
st.write(f"Mean$^2$ + Variance: {sum_formula}")
st.write(f"Sample Covariance Matrix:\n{cov_sample}")

st.latex(r"\frac{1}{n} \sum_{i=1}^n x_{i,d}^2 = (\bar{x}_d)^2 + \mathrm{Var}(x_d)")

st.write(f"Difference (dim 1): {abs(mse[0] - sum_formula[0]):.3e}")
st.write(f"Difference (dim 2): {abs(mse[1] - sum_formula[1]):.3e}")

# Plot histogram for each dimension and scatter
fig, axs = plt.subplots(1, 3, figsize=(18, 4))

axs[0].hist(x[:,0], bins=30, color='skyblue', edgecolor='k', alpha=0.7, label='Samples dim 1')
axs[0].axvline(mean_x[0], color='red', linestyle='--', label=f"Mean = {mean_x[0]:.2f}")
axs[0].axvline(mean_x[0] + np.sqrt(var_x[0]), color='green', linestyle=':', label=f"Mean + Std")
axs[0].axvline(mean_x[0] - np.sqrt(var_x[0]), color='green', linestyle=':', label=f"Mean - Std")
axs[0].set_title("Histogram of Samples (dim 1)")
axs[0].set_xlabel("x₁")
axs[0].set_ylabel("Frequency")
axs[0].legend()

axs[1].hist(x[:,1], bins=30, color='orange', edgecolor='k', alpha=0.7, label='Samples dim 2')
axs[1].axvline(mean_x[1], color='red', linestyle='--', label=f"Mean = {mean_x[1]:.2f}")
axs[1].axvline(mean_x[1] + np.sqrt(var_x[1]), color='green', linestyle=':', label=f"Mean + Std")
axs[1].axvline(mean_x[1] - np.sqrt(var_x[1]), color='green', linestyle=':', label=f"Mean - Std")
axs[1].set_title("Histogram of Samples (dim 2)")
axs[1].set_xlabel("x₂")
axs[1].set_ylabel("Frequency")
axs[1].legend()

axs[2].scatter(x[:,0], x[:,1], alpha=0.6, color='purple', edgecolor='k')
axs[2].set_title("Scatter Plot of Samples")
axs[2].set_xlabel("x₁")
axs[2].set_ylabel("x₂")
axs[2].axvline(mean_x[0], color='red', linestyle='--', alpha=0.5)
axs[2].axhline(mean_x[1], color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
st.pyplot(fig)