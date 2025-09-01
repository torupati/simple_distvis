import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.title("Random Walk Simulation")
seed = st.number_input("Set random seed", min_value=0, max_value=999999, value=0, step=1)
steps = st.number_input("Number of steps", min_value=10, max_value=100, value=50, step=10)
speed = 0.2

if st.button("Run Simulation"):
    x = np.zeros(steps + 1) # position time series
    x_prob = np.zeros(2 * steps + 1) # probability distribution
    x_prob[steps] = 1.0 # corresponds to index = 0

    placeholder = st.empty()
    for i in range(1, steps):
        dx = np.random.choice([-1, 1])
        x[i] = x[i-1] + dx

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        ax1.plot(np.arange(i+1), x[:i+1], marker='o')
        ax1.set_xlim(0, steps)
        ax1.set_title("Random Walk Path")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Y Position")
        ax1.set_ylim(-steps, steps)
        ax1.grid(True)
    
        new_prob = np.zeros_like(x_prob)
        for j in np.arange(1, len(x_prob)-1):
            new_prob[j] = 0.5 * (x_prob[j-1] + x_prob[j+1])
        x_prob = new_prob
        _x_prob_pos = np.arange(-steps, steps + 1)
        ax2.step(_x_prob_pos, x_prob, where='mid', color='orange')
        ax2.fill_between(_x_prob_pos, x_prob, color='orange', alpha=0.3)
        ax2.axvline(x[i], color='blue', linestyle='--', label='Current Position')
        ax2.legend()
        ax2.set_xlim(-steps, steps)
        ax2.set_title("Probability Distribution")
        ax2.set_xlabel("X Position")
        ax2.set_yticks([])
        ax2.grid(True)

        fig.tight_layout()
        placeholder.pyplot(fig)
        #st.pyplot(fig)
        plt.close(fig)
        time.sleep(0.1)
    st.success("Simulation complete!")