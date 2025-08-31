import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile
import logging

logging.basicConfig(level=logging.DEBUG)

st.set_page_config(layout="wide")

st.title("Diffusion Process")

seed = st.number_input("Set random seed", min_value=0, max_value=999999, value=0, step=1)

def run_simulation(seed):
    np.random.seed(seed)
    x = np.linspace(-2, 2, 400)
    timesteps: int = 50
    initial_std = 0.1
    vel_sig2 = 0.01
    pos_min, pos_max = -2, 2
    samples = []
    times = []

    fig, (ax1, ax2) = plt.subplots(1, 2)
    line, = ax1.plot([], [], lw=2)
    fill_area = ax1.fill_between([], [], color='orange', alpha=0.3)
    print(f"{fill_area=}")
    ax1.set_xlim(pos_min, pos_max)
    ax1.set_ylim(0, 5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("Probability Density")

    scat = ax2.scatter([], [], color='blue', s=30)
    line2, = ax2.plot([], [], color='orange', lw=2)
    ax2.set_xlim(0, timesteps)
    ax2.set_ylim(pos_min, pos_max)
    ax2.set_xlabel("Time step t")
    ax2.set_ylabel("Sampled position x")
    ax2.set_title("Sampled position x vs time t")

    def gaussian(x, mean, var):
        return 1/(np.sqrt(2*np.pi*var)) * np.exp(-(x-mean)**2/(2*var))

    def init():
        logging.debug("init")
        nonlocal samples, times
        line.set_data([], [])
        line2.set_data([], [])
        _x = np.random.normal(0, initial_std)
        samples = [_x]
        times = [0]
        scat.set_offsets(np.column_stack(([], [])))
        return line, scat, line2

    def update(frame:int):
        nonlocal samples, times, fill_area
        sig2 = initial_std**2 + frame * vel_sig2
        sigma = np.sqrt(sig2)

        _dx = np.random.normal(0, np.sqrt(vel_sig2))
        _x = samples[-1] + _dx if samples else 0
        samples.append(_x)
        times.append(frame)
        logging.debug(f"step={frame}, sample={_x:.3f}, var={sig2:.3f}")

        y = gaussian(x, 0, sig2)
        line.set_data(x, y)
        ax1.set_title(r"step = {frame}, $\sigma^2$ = {sig2:.2f}".format(frame=frame, sig2=sig2))
        ax1.grid(True)

        # Fill one-sigma area for current frame
        mask = (x >= -sigma) & (x <= sigma)
        fill_area = np.column_stack((x[mask], y[mask]))

        data = np.column_stack((times[:frame+1], samples[:frame+1]))
        scat.set_offsets(data)
        line2.set_data(times[:frame+1], samples[:frame+1])
        ax2.set_title(f"Sampled position x vs time t (t={frame})")
        return line, scat, line2

    ani = FuncAnimation(fig, update, frames=timesteps, init_func=init, blit=True)

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
        ani.save(tmpfile.name, writer="pillow", fps=2)
        st.image(tmpfile.name)

if st.button("Recalculate Simulation"):
    run_simulation(seed)
else:
    run_simulation(seed)