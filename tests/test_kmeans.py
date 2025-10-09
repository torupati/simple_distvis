import numpy as np
import pytest

from hmm.kmeans import KmeansCluster
from src.hmm.kmeans import kmeans_clustering


class TestKmeansCluster:
    def test_kl_divergence(self):
        x = np.array([0.5, 0.3, 0.2])
        y = np.array([0.4, 0.4, 0.2])
        result = KmeansCluster.KL_divergence(x, y)
        assert isinstance(result, float)
        assert result >= 0

    def test_invalid_train_mode(self):
        cluster = KmeansCluster(3, 2)
        cluster._train_mode = "INVALID"
        with pytest.raises(
            NotImplementedError, match="Only TRAIN_VAR_INSIDE is supported now"
        ):
            cluster.UpdateParameters()


def test_kmeans_constructor():
    try:
        _ = KmeansCluster(10, 10, covariance_mode="diag")
    except ValueError as e:
        print(e)
    try:
        _ = KmeansCluster(10, 10, covariance_mode="full")
    except ValueError as e:
        print(e)
    try:
        _ = KmeansCluster(10, 10, covariance_mode="none")
    except ValueError as e:
        print(e)
    try:
        _ = KmeansCluster(10, 10, covariance_mode="foofoo")
    except ValueError as e:
        print(e)


def test_kmeans_fit_predict():
    np.random.seed(0)
    n_samples = 300
    n_features = 5
    n_clusters = 3
    X = np.random.rand(n_samples, n_features)
    print(f"{X.shape=}")

    mu_init = np.random.rand(n_clusters, n_features)
    kmeans, cost_history = kmeans_clustering(X, mu_init)
    for i, c in enumerate(cost_history):
        print(f"itr={i} cost={c}")
        assert isinstance(c, float)
        assert c >= 0
        if i > 1:
            assert cost_history[i] <= cost_history[i - 1]
    labels = kmeans.predict(X)
    assert labels.shape == (n_samples,)
    assert set(labels) <= set(range(n_clusters))
