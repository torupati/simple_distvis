import io
import json
import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


st.set_page_config(layout="centered")  # centered or wide

st.title("2D Visualization and GMM EM Algorithm")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "txt"])

data = None

if uploaded_file is not None:
    try:
        data = np.loadtxt(uploaded_file, delimiter=",", comments="#")
        st.write("Shape of data:", data.shape)
    except ValueError as e:
        st.error(f"Error loading file: {e}")
        data = None

if data is not None:
    # 2D Scatter plot of first two columns
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], alpha=0.7, label="Data")
    ax.set_xlabel("Column 1")
    ax.set_ylabel("Column 2")
    ax.set_title("2D Scatter Plot (Col 1 vs Col 2)")

    # Button to show mean and covariance
    if st.button("Show Mean and Covariance"):
        mean = np.mean(data[:, :2], axis=0)
        cov = np.cov(data[:, :2].T)
        # Plot mean
        ax.scatter(mean[0], mean[1], color="red", marker="x", s=100, label="Mean")
        # Plot covariance ellipse
        from matplotlib.patches import Ellipse

        def plot_cov_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * n_std * np.sqrt(vals)
            ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)
            ax.add_patch(ellip)

        plot_cov_ellipse(
            mean,
            cov,
            ax,
            n_std=2,
            edgecolor="red",
            facecolor="none",
            lw=2,
            label="Covariance (2σ)",
        )
        ax.legend()

    st.pyplot(fig)

    # GMM EM algorithm
    n_mixtures = st.number_input(
        "Number of Gaussian", min_value=1, max_value=10, value=2, step=1
    )

    st.subheader("EM Algorithm of Gaussian Mixture Models (in-house implementation)")
    if st.button("Run GMM Training (in-house)"):
        import pickle

        from src.hmm.gmm import GaussianMixtureModel, train_gmm
        from src.hmm.kmeans_plot import plot_data_with_centroid

        data = data[:, [0, 1]]
        mu_init = np.random.randn(n_mixtures, data.shape[1])  # Random initialization
        gmm = GaussianMixtureModel(M=n_mixtures, D=data.shape[1])
        gmm.Mu = mu_init
        gmmparam, loglikelihood_history = train_gmm(
            gmm, data, max_it=20, plot_ckpt=False
        )
        print(gmmparam)
        _gamma, _ll = gmm.update_e_step(data)
        mu = gmmparam.Mu
        fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
        plot_data_with_centroid(ax3, data, _gamma, mu)
        ax3.set_title(f"GMM In-house (m={n_mixtures})")
        st.session_state["gmm_fig3"] = fig3  # Save to session

        fig3b, ax3b = plt.subplots(1, 1, figsize=(6, 4))
        from src.hmm.gmm import plot_loglikelihood_history

        ax3b = plot_loglikelihood_history(ax3b, loglikelihood_history)
        ax3b.set_title(f"GMM In-house Log-Likelihood History (m={n_mixtures})")
        st.session_state["gmm_fig3b"] = fig3b  # Save to session

        # --- Download GMM parameters as a dictionary ---
        gmm_dict = {
            "weights": gmmparam.Pi.tolist(),
            "means": gmmparam.Mu.tolist(),
            "covariances": gmmparam.Sigma.tolist(),
            "loglikelihood_history": loglikelihood_history,
        }

        st.session_state["gmm_dict"] = gmm_dict  # Save to session

    st.subheader("GMM Clustering (scikit-learn)")
    if st.button("Run GMM Clustering (scikit-learn)"):
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=n_mixtures, covariance_type="full", random_state=0
        )
        labels = gmm.fit_predict(data[:, :2])
        centers = gmm.means_

        fig4, ax4 = plt.subplots()
        scatter = ax4.scatter(
            data[:, 0], data[:, 1], c=labels, cmap="tab10", alpha=0.7, label="Data"
        )
        ax4.scatter(
            centers[:, 0],
            centers[:, 1],
            color="red",
            marker="x",
            s=150,
            label="Centers",
        )
        ax4.set_xlabel("Column 1")
        ax4.set_ylabel("Column 2")
        ax4.set_aspect("equal", "box")
        ax4.set_title(f"GMM Clustering (scikit-learn) (m={n_mixtures})")
        ax4.legend()
        st.pyplot(fig4)

# --- Download UI & Figure display ---
if "gmm_dict" in st.session_state:
    # Show figures if available
    if "gmm_fig3" in st.session_state:
        st.pyplot(st.session_state["gmm_fig3"])
    if "gmm_fig3b" in st.session_state:
        st.pyplot(st.session_state["gmm_fig3b"])

    st.markdown("### Download GMM Parameters")
    file_format = st.radio("Select file format", ("pickle", "json"))
    gmm_dict = st.session_state["gmm_dict"]

    if file_format == "pickle":
        buffer = io.BytesIO()
        pickle.dump(gmm_dict, buffer)
        buffer.seek(0)
        st.download_button(
            label="Download GMM Parameters (pickle)",
            data=buffer,
            file_name="gmm_parameters.pkl",
            mime="application/octet-stream",
        )
    else:  # json
        json_str = json.dumps(gmm_dict, indent=4)
        st.download_button(
            label="Download GMM Parameters (json)",
            data=json_str,
            file_name="gmm_parameters.json",
            mime="application/json",
        )
