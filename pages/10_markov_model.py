import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Markov Model State Probability Evolution")

# User input for number of states
n_states = st.number_input("Number of states", min_value=2, max_value=10, value=3, step=1)

# Generate probability button
if st.button("Generate probability"):
    init_prob = np.random.rand(n_states)
    init_prob /= init_prob.sum()
    trans_mat = np.random.rand(n_states, n_states)
    trans_mat /= trans_mat.sum(axis=1, keepdims=True)
    st.session_state["init_prob"] = init_prob
    st.session_state["trans_mat"] = trans_mat
    st.success("Initial probability and transition matrix generated!")

# Editable initial probability and transition matrix
if "init_prob" in st.session_state and "trans_mat" in st.session_state:
    st.write("Edit the initial probability and transition matrix below. (Rows of transition matrix should sum to 1)")
    # Initial probability
    init_prob = st.session_state["init_prob"]
    try:
        init_prob_edit = st.text_input(
            "Initial probability (comma separated)", 
            value=",".join([f"{v:.4f}" for v in init_prob])
        )
        init_prob_new = np.array([float(x) for x in init_prob_edit.split(",")])
        if len(init_prob_new) == n_states and np.all(init_prob_new >= 0):
            init_prob_new /= init_prob_new.sum()
            st.session_state["init_prob"] = init_prob_new
        else:
            st.warning("Please enter non-negative values, one for each state.")
    except ValueError as e:
        st.warning("Invalid input for initial probability.")

    # Transition matrix
    trans_mat = st.session_state["trans_mat"]
    trans_mat_str = "\n".join([",".join([f"{v:.4f}" for v in row]) for row in trans_mat])
    trans_mat_edit = st.text_area(
        "Transition matrix (each row comma separated, rows separated by newlines)",
        value=trans_mat_str,
        height=150
    )
    try:
        rows = [list(map(float, row.split(","))) for row in trans_mat_edit.strip().split("\n")]
        trans_mat_new = np.array(rows)
        if trans_mat_new.shape == (n_states, n_states) and np.all(trans_mat_new >= 0):
            # Normalize each row to sum to 1
            trans_mat_new = trans_mat_new / trans_mat_new.sum(axis=1, keepdims=True)
            st.session_state["trans_mat"] = trans_mat_new
        else:
            st.warning("Please enter non-negative values for a square matrix.")
    except ValueError:
        st.warning("Invalid input for transition matrix. Ensure all values are numeric.")
    except Exception as e:
        st.warning(f"An unexpected error occurred: {e}")

    st.write("Current initial probability:", st.session_state["init_prob"])
    st.write("Current transition matrix:")
    st.write(st.session_state["trans_mat"])
    run_enabled = True
else:
    run_enabled = False

# Run simulation button (only enabled if probabilities exist)
if st.button("Run simulation", disabled=not run_enabled):
    n_steps = 20
    probs = np.zeros((n_steps, n_states))
    probs[0] = st.session_state["init_prob"]
    for t in range(1, n_steps):
        probs[t] = probs[t-1] @ st.session_state["trans_mat"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(n_states):
        ax.plot(range(n_steps), probs[:, i], label=f"State {i}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.set_title("State Probabilities Over Time")
    ax.legend()
    ax.set_ylim(bottom=0)  # Set Y-axis minimum to 0
    ax.set_xlim(xmin=0, xmax=n_steps - 1)  # Set X-axis limits
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Make X-axis labels integer
    st.pyplot(fig)

    # Show final state probabilities as a table
    st.markdown("### Final State Probabilities")
    final_probs = probs[-1]
    st.table({f"State {i}": [final_probs[i]] for i in range(n_states)})