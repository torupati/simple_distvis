from src.hmm import foo
import pickle
import argparse

import numpy as np

import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

import streamlit as st
import pickle

st.title("K-means Clustering")

uploaded_file = st.file_uploader("Upload Pickle file from here.", type=["pkl", "pickle"])

if uploaded_file is not None:
    data = pickle.load(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("keys:", list(data.keys()))
    if 'sample' in data:
        st.write("sampleのshape:", data['sample'].shape)
