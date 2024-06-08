import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit as st
from streamlit.components.v1 import html
from matplotlib.gridspec import GridSpec


N_L, N_R, S, p = [], [], [], []

def silnia(n):
    s = 1
    n = int(n)
    for i in range(2, n+1):
        s *= i
    return s

def pos_init(N):
    posx = np.random.randint(5, size=N)
    posy = np.random.randint(10, size=N)
    pos = np.array([posx, posy])
    return pos

def dir_init(N):
    dir = np.random.choice(4, N)  # Assuming directions up down left right 0,1,2,3
    return dir

def update_positions(pos, dir, grid_size):
    for i in range(pos.shape[1]):
        dir[i] = np.random.choice(4)  # Random direction at each step

        if dir[i] == 0:  # Up
            pos[1, i] += 1
        elif dir[i] == 1:  # Down
            pos[1, i] -= 1
        elif dir[i] == 2:  # Left
            pos[0, i] -= 1
        elif dir[i] == 3:  # Right
            pos[0, i] += 1

        # Reflecting off walls
        if pos[0, i] < 0:
            pos[0, i] = 0
        elif pos[0, i] >= grid_size:
            pos[0, i] = grid_size - 1

        if pos[1, i] < 0:
            pos[1, i] = 0
        elif pos[1, i] >= grid_size:
            pos[1, i] = grid_size - 1

def animate_particle_movement(pos, dir, grid_size, total_steps, dt):
    global N_R, N_L, S, p
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1], width_ratios=[2, 1])

    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[1:, 1])

    ax0.set_xlim(0, grid_size)
    ax0.set_ylim(0, grid_size)
    ax0.set_xticks(np.arange(0, grid_size + 1, 1))
    ax0.set_yticks(np.arange(0, grid_size + 1, 1))
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    ax0.grid(True)

    N = pos.shape[1]
    particles = ax0.scatter(pos[0, :], pos[1, :], color='purple')

    def update(frame):
        global N_R, N_L, S, p
        nonlocal pos, dir, particles
        update_positions(pos, dir, grid_size)
        particles.set_offsets(pos.T)  # Update particle positions

        num_right = np.sum(pos[0, :] > grid_size/2)
        num_left = np.sum(pos[0, :] <= grid_size/2)
        N_L.append(num_left)
        N_R.append(num_right)

        S_value = np.log(silnia(N) / (silnia(num_left) * silnia(num_right)))
        S.append(S_value)

        p_value = (num_left - num_right) / N
        p.append(p_value)

        ax1.clear()
        ax1.plot(N_L, "-", color="green", label="N_L")
        ax1.plot(N_R, "-", color="red", label="N_R")
        ax1.set_xlim(0, total_steps / dt)
        ax1.set_ylim(-0.5, N+0.5)
        ax1.axhline(y= N / 2, color='purple', linestyle='--', label="N/2")
        ax1.legend()

        ax2.clear()
        ax2.plot(p, "-", color="green", label=r"$\frac{N_L - N_R}{N}$")
        ax2.axhline(y= 0, color='purple', linestyle='--')
        ax2.set_xlim(0, total_steps / dt)
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel("p")
        ax2.legend()

        ax3.clear()
        ax3.plot(S, "-", color="red", label = r"$\log(\frac{N!}{N_R!N_L!})$")
        ax3.set_xlim(0, total_steps / dt)
        ax3.set_ylim(0, np.log(silnia(N) / (silnia(N/2) * silnia(N/2)))+1)
        ax3.axhline(y=np.log(silnia(N) / (silnia(N/2) * silnia(N/2))), color='purple', linestyle='--', label=r"$N_{L} = \frac{N}{2}$")
        ax3.set_ylabel(r"$\sigma $")
        ax3.legend()

        return particles,

    plt.rcParams['animation.embed_limit'] = 100  # Increase the embed limit to 100MB
    ani = FuncAnimation(fig, update, frames=np.arange(0, total_steps, dt), blit=True, repeat=False)

    # Convert animation to HTML string
    animation_html = ani.to_jshtml()

    return animation_html

# Streamlit app
st.title("Grid Gas Simulation")
st.markdown("""
    <style>
        .main .block-container {
            max-width: 60%;
            padding: 2rem;
        }
        .css-1v3fvcr {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

N = st.sidebar.slider("Number of particles", min_value=1, max_value=100, value=10)
grid_size = st.sidebar.slider("Grid size", min_value=5, max_value=20, value=10)
total_steps = st.sidebar.slider("Total steps", min_value=50, max_value=500, value=100)
dt = st.sidebar.slider("Time step (dt)", min_value=1, max_value=10, value=1)

poss = pos_init(N)
dirs = dir_init(N)

if st.sidebar.button("Run Simulation"):
    animation_html = animate_particle_movement(poss, dirs, grid_size, total_steps, dt)
    html(animation_html, height=1000)