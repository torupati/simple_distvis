import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Normal Distribution 1D")

# User input for mean and variance
mean = st.slider("Mean", -10.0, 10.0, 2.0, step=0.1)
variance = st.slider("Variance", 0.1, 100.0, 2.0, step=0.1)
show_grid = st.checkbox("show grid", value=False)
fix_xlim = st.checkbox("fix x limit to (-10, 10)", value=True)
erase_xticks = st.checkbox("erase x tick labels", value=False) 
x0_line = st.checkbox("show x=0 line", value=False)

def normal_pdf(x, mu, sigma2):
    """Calculate normal distribution density function

    Args:
        x (float): _input value
        mu (float): _mean value
        sigma2 (flaot): _variance value

    Returns:
        _type_: _description_
    """
    return 1.0 / np.sqrt(2 * np.pi * sigma2) * np.exp(- (x - mu) ** 2 / (2 * sigma2))

def plot_normal_1d(mu, sigma2):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(mu - 4 * np.sqrt(sigma2), mu + 4 * np.sqrt(sigma2), 100)
    pdf = normal_pdf(x, mu, sigma2)
    ax.plot(x, pdf, label=r"P(X=x)", color='blue')
    ax.fill_between(x, pdf, color='blue', alpha=0.1)
    ax.axvline(mu, color='red', linestyle='--')
    ax.axvline(mu + np.sqrt(sigma2), color='green', linestyle='-', linewidth=0.8)
    ax.axvline(mu - np.sqrt(sigma2), color='green', linestyle='-', linewidth=0.8)
    ax.fill_betweenx([0, np.max(pdf)*1.1], x1=mu - np.sqrt(sigma2), x2=mu + np.sqrt(sigma2), color='green', alpha=0.1)
    ax.set_ylim(0, np.max(pdf) * 1.1)
    ax.set_ylim(0, None)
    ax.set_title("1D Gaussian Distribution")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("Probability Density")
    ax.legend()
    return fig, ax

# Plot 1D Gaussian distribution with sample mean and variance
fig2, ax2 = plot_normal_1d(mean, variance)
if show_grid:
    ax2.grid(True)
if fix_xlim:
    ax2.set_xlim(-10, 10)
if erase_xticks:
    ax2.set_xticklabels([])
if x0_line:
    ax2.axvline(0, color='black', linestyle='solid', alpha=0.3)

st.pyplot(fig2)

def plot_normal_1d_double(mu, sigma2):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(mu - 8 * np.sqrt(sigma2), mu + 8 * np.sqrt(sigma2), 400)
    pdf = normal_pdf(x, mu, sigma2)
    ax.plot(x, pdf, label=r"P(X=x)", color='black')
    #ax.fill_between(x, pdf, color='blue', alpha=0.1)
    ax.axvline(mu, color='black', linestyle='dotted')

    pdf_double = normal_pdf(x, 2*mu, 4*sigma2)
    ax.plot(x, pdf_double, label=r"$P(X_1+X_2=x)$", color='green')
    ax.fill_between(x, pdf_double, color='green', alpha=0.1)
    #ax.axvline(mu*2 + np.sqrt(4*sigma2), color='green', linestyle='-', linewidth=0.8)
    #ax.axvline(mu*2 - np.sqrt(4*sigma2), color='green', linestyle='-', linewidth=0.8)
    ax.axvline(2*mu, color='red', linestyle='--')

    pdf_cancel = normal_pdf(x, mu-mu, 4*sigma2)
    ax.plot(x, pdf_cancel, label=r"$P(X_1-X_2=x)$", color='green')
    ax.fill_between(x, pdf_cancel, color='blue', alpha=0.1)
    #ax.axvline(mu*2 + np.sqrt(4*sigma2), color='green', linestyle='-', linewidth=0.8)
    #ax.axvline(mu*2 - np.sqrt(4*sigma2), color='green', linestyle='-', linewidth=0.8)
    ax.axvline(mu-mu, color='red', linestyle='--')

    ymax = ax.get_ylim()[1]
    ax.fill_betweenx([0, ymax], x1=2*mu - 2*np.sqrt(sigma2),
                     x2=2*mu + 2*np.sqrt(sigma2), color='cyan', alpha=0.1)
    ax.set_ylim(0, ymax)

    #ax.set_title("1D Gaussian Distribution")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("Probability Density")
    return fig, ax

fig_double, ax_double = plot_normal_1d_double(mean, variance)
if show_grid:
    ax_double.grid(True)
if fix_xlim:
    ax_double.set_xlim(-10, 10)
if erase_xticks:
    ax_double.set_xticklabels([])
if x0_line:
    ax_double.axvline(0, color='black', linestyle='solid', alpha=0.3)
ax_double.legend()
st.pyplot(fig_double)

def plot_normal_1d_average_of_tow(mu, sigma2):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(mu - 8 * np.sqrt(sigma2), mu + 8 * np.sqrt(sigma2), 400)
    pdf = normal_pdf(x, mu, sigma2)
    ax.plot(x, pdf, label=r"$P(X=x)$", color='gray')

    pdf_double = normal_pdf(x, 2*mu, 4*sigma2)
    ax.plot(x, pdf_double, label=r"$P(X_1+X_2=x)$", color='green')
    #ax.fill_between(x, pdf_double, color='green', alpha=0.1)

    pdf_average = normal_pdf(x, mu, 0.5*sigma2)
    ax.plot(x, pdf_average, label=r"$P(\frac{X_1+X_2}{2}=x)$", color='blue')
    ax.fill_between(x, pdf_average, color='blue', alpha=0.1)

    #ax.axvline(mu*2 + np.sqrt(4*sigma2), color='green', linestyle='-', linewidth=0.8)
    #ax.axvline(mu*2 - np.sqrt(4*sigma2), color='green', linestyle='-', linewidth=0.8)
    ax.axvline(mu, color='red', linestyle='--')
    ymax = ax.get_ylim()[1]
    #ax.fill_betweenx([0, ymax], x1=mu - np.sqrt(sigma2),
    #                 x2=mu + np.sqrt(sigma2), color='green', alpha=0.1)
    #ax.fill_betweenx([0, ymax], x1=2*mu - 2*np.sqrt(sigma2),
    #                 x2=2*mu + 2*np.sqrt(sigma2), color='cyan', alpha=0.1)
    ax.fill_betweenx([0, ymax], x1=mu - np.sqrt(sigma2),
                     x2=mu + np.sqrt(sigma2), color='cyan', alpha=0.1)
    ax.set_ylim(0, ymax)
    #ax.set_title("1D Gaussian Distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")
    return fig, ax

fig_ave, ax_ave = plot_normal_1d_average_of_tow(mean, variance)
if show_grid:
    ax_ave.grid(True)
if fix_xlim:
    ax_ave.set_xlim(-10, 10)
if erase_xticks:
    ax_ave.set_xticklabels([])
if x0_line:
    ax_ave.axvline(0, color='black', linestyle='solid', alpha=0.3)
ax_ave.legend()
st.pyplot(fig_ave)
