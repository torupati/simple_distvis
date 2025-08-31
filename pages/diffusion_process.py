import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile

st.title("Diffusion Process")

x = np.linspace(-2, 2, 400)
timesteps = 50
initial_std = 0.1

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(-2, 2)
ax.set_ylim(0, 5)
ax.set_xlabel("x")
ax.set_ylabel("Probability Density")

def gaussian(x, mean, var):
    return 1/(np.sqrt(2*np.pi*var)) * np.exp(-(x-mean)**2/(2*var))

def init():
    line.set_data([], [])
    return line,

def update(frame):
    std = initial_std**2 + frame * 0.01
    y = gaussian(x, 0, std)
    line.set_data(x, y)
    ax.set_title(f"t = {frame}, stdev = {std:.2f}")
    return line,

ani = FuncAnimation(fig, update, frames=timesteps, init_func=init, blit=True)

with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
    ani.save(tmpfile.name, writer="pillow", fps=2)
    st.image(tmpfile.name)