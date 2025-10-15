import io
import pickle

import numpy as np
import streamlit as st

from src.hmm.sampler import generate_gmm_samples, generate_sample_parameter

st.title("GMM Sample Generator")

# Input fields
num_mixture = st.number_input(
    "Number of mixtures", min_value=1, max_value=20, value=3, step=1
)
feature_dim = st.number_input(
    "Feature dimension", min_value=1, max_value=10, value=2, step=1
)
n_sample = st.number_input(
    "Number of samples", min_value=1, max_value=10000, value=100, step=1
)

if st.button("Generate samples"):
    # Generate GMM parameters
    weights, mean_vectors, covariances = generate_sample_parameter(
        num_mixture, feature_dim
    )
    samples, labels = generate_gmm_samples(n_sample, weights, mean_vectors, covariances)
    # Save to session_state
    st.session_state["gmm_samples"] = {
        "samples": samples,
        "labels": labels,
        "weights": weights,
        "means": mean_vectors,
        "covariances": covariances,
    }

# Show results and download UI if data exists in session_state
if "gmm_samples" in st.session_state:
    data = st.session_state["gmm_samples"]
    st.write("Mixture weights:", data["weights"])
    st.write("Mean vectors:", data["means"])
    covariances = data["covariances"]
    if len(covariances.shape) == 2:
        st.write("Covariances (diagonal):", covariances)
    else:
        for k in range(len(data["weights"])):
            st.write(f"Covariance matrix for mixture {k}:\n", covariances[k])
            eigvals = np.linalg.eigvalsh(covariances[k])
            if np.any(eigvals <= 0):
                st.warning(
                    f"Covariance matrix for mixture {k} is not positive definite. Eigenvalues: {eigvals}"
                )
            else:
                st.write(
                    f"Covariance matrix for mixture {k} is positive definite. Eigenvalues: {eigvals}"
                )

    st.write("Generated samples shape:", data["samples"].shape)
    st.write("First 5 samples:", data["samples"][:5])
    st.write("First 5 labels:", data["labels"][:5])

    st.markdown("### Download samples")
    file_format = st.radio("Select file format", ("pickle", "csv", "csv(with labels)"))

    if file_format == "pickle":
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)
        st.download_button(
            label="Download samples (pickle)",
            data=buffer,
            file_name="gmm_samples.pkl",
            mime="application/octet-stream",
        )
    elif file_format == "csv":
        csv_buffer = io.StringIO()
        header = ",".join([f"feature_{i + 1}" for i in range(data["samples"].shape[1])])
        np.savetxt(
            csv_buffer,
            data["samples"],
            delimiter=",",
            header=f"#{header}",  # Add # at the beginning
            comments="",
        )
        st.download_button(
            label="Download samples (csv)",
            data=csv_buffer.getvalue(),
            file_name="gmm_samples.csv",
            mime="text/csv",
        )
    else:  # csv(with labels)
        csv_data = np.hstack([data["samples"], data["labels"].reshape(-1, 1)])
        header = ",".join(
            [f"feature_{i + 1}" for i in range(data["samples"].shape[1])] + ["label"]
        )
        csv_buffer = io.StringIO()
        np.savetxt(
            csv_buffer,
            csv_data,
            delimiter=",",
            header=f"#{header}",  # Add # at the beginning
            comments="",
        )
        st.download_button(
            label="Download samples (csv)",
            data=csv_buffer.getvalue(),
            file_name="gmm_samples.csv",
            mime="text/csv",
        )
