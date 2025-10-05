import pytest
import numpy as np
from src.hmm.kmeans import KmeansCluster

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
        with pytest.raises(NotImplementedError, match="Only TRAIN_VAR_INSIDE is supported now"):
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