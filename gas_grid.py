import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit as st
from streamlit.components.v1 import html

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
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    N = pos.shape[1]
    particles = ax.scatter(pos[0, :], pos[1, :], color='purple')

    def update(frame):
        nonlocal pos, dir, particles
        update_positions(pos, dir, grid_size)
        particles.set_offsets(pos.T)  # Update particle positions
        return particles,

    ani = FuncAnimation(fig, update, frames=np.arange(0, total_steps, dt), blit=True, repeat=False)

    # Convert animation to HTML string
    animation_html = ani.to_jshtml()

    return animation_html

# Streamlit app
st.title("Particle Movement Simulation")

N = st.sidebar.slider("Number of particles", min_value=1, max_value=100, value=10)
grid_size = st.sidebar.slider("Grid size", min_value=5, max_value=20, value=10)
total_steps = st.sidebar.slider("Total steps", min_value=50, max_value=500, value=100)
dt = st.sidebar.slider("Time step (dt)", min_value=1, max_value=10, value=1)

poss = pos_init(N)
dirs = dir_init(N)

if st.sidebar.button("Run Simulation"):
    animation_html = animate_particle_movement(poss, dirs, grid_size, total_steps, dt)
    html(animation_html, height=600)