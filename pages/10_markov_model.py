import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Markov Model State Probability Evolution")

# User input for number of states
n_states = st.number_input("Number of states", min_value=2, max_value=10, value=3, step=1)

if st.button("Generate and Simulate"):
    # Generate random initial probability (sum to 1)
    init_prob = np.random.rand(n_states)
    init_prob /= init_prob.sum()
    st.write("Initial probability:", init_prob)

    # Generate random transition matrix (rows sum to 1)
    trans_mat = np.random.rand(n_states, n_states)
    trans_mat /= trans_mat.sum(axis=1, keepdims=True)
    st.write("Transition matrix:")
    st.write(trans_mat)

    # Simulate state probabilities over 20 steps
    n_steps = 20
    probs = np.zeros((n_steps, n_states))
    probs[0] = init_prob
    for t in range(1, n_steps):
        probs[t] = probs[t-1] @ trans_mat

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(n_states):
        ax.plot(range(n_steps), probs[:, i], label=f"State {i}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.set_title("State Probabilities Over Time")
    ax.legend()
    st.pyplot(fig)