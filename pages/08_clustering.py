# pages/10_kmeans_clustering.py

import logging

import numpy as np

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

import matplotlib.pyplot as plt
import streamlit as st

# st.set_page_config(layout="wide")
st.set_page_config(layout="centered")
st.title("2D Scatter Plot from CSV")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "txt"])

data = None

if uploaded_file is not None:
    try:
        data = np.loadtxt(uploaded_file, delimiter=",", comments="#")
        st.write("Shape of data:", data.shape)
    except Exception as e:
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

    # K-means clustering controls
    n_clusters = st.number_input(
        "Number of clusters", min_value=1, max_value=10, value=2, step=1
    )

    # K-means clustering using scikit-learn
    st.subheader("K-means Clustering (scikit-learn)")
    if st.button("Run K-means Clustering(scikit-learn)"):
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(data[:, :2])
        centers = kmeans.cluster_centers_

        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(
            data[:, 0], data[:, 1], c=labels, cmap="tab10", alpha=0.7, label="Data"
        )
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            color="red",
            marker="x",
            s=150,
            label="Centers",
        )
        ax2.set_xlabel("Column 1")
        ax2.set_ylabel("Column 2")
        ax2.set_title(f"K-means Clustering (k={n_clusters})")
        ax2.legend()
        st.pyplot(fig2)

    # K-means clustering in-house implementation (very basic)
    st.subheader("K-means Clustering (in-house implementation)")
    if st.button("Run K-means Clustering (in-house)"):
        from src.hmm.kmeans import kmeans_clustering
        from src.hmm.kmeans_plot import plot_data_with_centroid

        data = data[:, [0, 1]]
        mu_init = np.random.randn(n_clusters, data.shape[1])  # Random initialization
        kmeansparam, cost_history = kmeans_clustering(data, mu_init)
        print(kmeansparam)
        _r = kmeansparam.get_alignment(x=data)
        mu = kmeansparam.Mu
        fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
        plot_data_with_centroid(ax3, data, _r, mu)
        ax3.set_title(f"K-means Clustering In-house (k={n_clusters})")
        st.pyplot(fig3)

        # kmeans_dict = {
