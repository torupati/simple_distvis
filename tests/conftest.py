import pytest
import numpy as np

@pytest.fixture
def sample_data():
    """Sample 2D data for clustering"""
    return np.random.randn(100, 2)

@pytest.fixture
def kmeans_instance():
    """Instance of KmeansCluster"""
    from src.hmm.kmeans import KmeansCluster
    return KmeansCluster(4, 3)
