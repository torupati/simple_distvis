from src.hmm import foo
import pickle
import argparse

import numpy as np

import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

import streamlit as st

st.title("K-means Clustering")


from src.hmm.kmeans import kmeans_clustering
from src.hmm.kmeans_plot import plot_distortion_history

def train_kmeans(input_file, num_cluster, random_seed, dist_mode):
    from tqdm import tqdm
    _logger.info("input: %s", input_file)
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
        #_logger.debug(data.keys())
        X = data['sample']
        _logger.info('model type: %s', data.get('model_type'))
        param = data['model_param']
        mu_init = param.Mu

    if num_cluster > 0:
        Dim = X.shape[1]
        np.random.seed(random_seed)
        mu_init = np.random.randn(num_cluster, Dim)
        _logger.info("initial points generated from randn (%d %d)", num_cluster, Dim)

    #X = np.abs(X + 1.0E-6)
    #mu_init = np.abs(mu_init + 1.0E-6)
    kmeansparam, cost_history = kmeans_clustering(X, mu_init,
                                                  dist_mode=dist_mode,
                                                  max_it = 100)
    #print('Mu:', kmeansparam.Mu)
    #print('Sigma:', kmeansparam.Sigma)

    out_pngfile = "distortion.png"
    fig = plot_distortion_history(cost_history)
    fig.savefig(out_pngfile)
    _logger.info('out: %s', out_pngfile)

    R = kmeansparam.get_alignment(X)
    out_file = 'out_kmeans.pickle'
    with open(out_file, 'wb') as f:
        pickle.dump({'model': kmeansparam,
                     'history': cost_history,
                     'iteration': len(cost_history),
                     'alignment': R},
                    f)
    #print(sum(R == 1))
    _logger.info('out: %s', out_file)


