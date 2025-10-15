import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.title("Square Error Decomposition: Mean Square = (Mean)$^2$ + Variance")

# User input for mean and variance
mean = st.slider("Mean", -10.0, 10.0, 2.0, step=0.1)
variance = st.slider("Variance", 0.1, 10.0, 2.0, step=0.1)
n = 200

# Generate random samples
np.random.seed(0)
x = np.random.normal(mean, np.sqrt(variance), n)

# Calculate statistics
mean_x = np.mean(x)
var_x = np.var(x)
mse = np.mean(x**2)
sum_formula = mean_x**2 + var_x

# Display results
st.write(f"Sample Mean: {mean_x:.3f}")
st.write(f"Sample Variance: {var_x:.3f}")
st.write(f"Mean of $x^2$ (MSE): {mse:.3f}")
st.write(f"Mean$^2$ + Variance: {sum_formula:.3f}")

# Check formula
st.latex(r"\frac{1}{n} \sum_{i=1}^n x_i^2 = (\bar{x})^2 + \mathrm{Var}(x)")
st.write(f"Difference: {abs(mse - sum_formula):.3e}")

# Plot histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(x, bins=30, color="skyblue", edgecolor="k", alpha=0.7, label="Samples")
ax.axvline(mean_x, color="red", linestyle="--", label=f"Mean = {mean_x:.2f}")
ax.axvline(mean_x + np.sqrt(var_x), color="green", linestyle=":", label="Mean + Std")
ax.axvline(mean_x - np.sqrt(var_x), color="green", linestyle=":", label="Mean - Std")
ax.set_title("Histogram of Samples")
ax.set_xlabel("x")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)
