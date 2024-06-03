import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import streamlit as st
import streamlit.components.v1 as components
from scipy.stats import rayleigh

# Initialize global lists for particles and velocities
V_R, V_L, N_L, N_R, S = [], [], [], [], []


def generuj_czastki(num_particles, min=0, max=1, rounding=2):
    positionsX = np.random.uniform(min, max, num_particles).round(rounding)
    positionsY = np.random.uniform(min, 1., num_particles).round(rounding)
    positions = np.vstack((positionsX, positionsY))
    return positions


def generate_velocity_matrix(num_particles, min_speed=-1, max_speed=1):
    return np.random.uniform(low=min_speed, high=max_speed, size=(2, num_particles))


def generate_velocity_matrix2(num_particles, positions, T_L, T_R):
    m = 1.67 * 10 ** (-27)
    k = 1.38 * 10 ** (-23)
    sigmaL = np.sqrt(k * T_L / m)
    sigmaR = np.sqrt(k * T_R / m)
    speedsL = 0.001 * rayleigh.rvs(scale=sigmaL, size=num_particles)
    speedsR = 0.001 * rayleigh.rvs(scale=sigmaR, size=num_particles)
    speeds = np.where(positions[0, :] >= 0.5, speedsR, speedsL)
    velocities = np.zeros(shape=(2, num_particles))
    angles = 2 * np.pi * np.random.rand(num_particles)
    velocities[0, :] = speeds * np.cos(angles)
    velocities[1, :] = speeds * np.sin(angles)
    return velocities


def bound_collision(N, positions, velocities):
    n_collision = 0
    for i in range(N):
        for j in range(2):
            if positions[j, i] <= 0 or positions[j, i] >= 1:
                velocities[j, i] *= -1
                n_collision += 1
    return velocities, n_collision


def collisions(positions):
    N = positions.shape[1]
    collision_matrix = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(positions[:, i] - positions[:, j]) < 0.02:
                collision_matrix[i, j] = collision_matrix[j, i] = True
    return collision_matrix


def resolve_collision(pos1, vel1, pos2, vel2):
    normal = pos2 - pos1
    normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]])

    vel1_normal = np.dot(vel1, normal)
    vel1_tangent = np.dot(vel1, tangent)
    vel2_normal = np.dot(vel2, normal)
    vel2_tangent = np.dot(vel2, tangent)

    vel1_normal, vel2_normal = vel2_normal, vel1_normal

    vel1_new = vel1_normal * normal + vel1_tangent * tangent
    vel2_new = vel2_normal * normal + vel2_tangent * tangent

    return vel1_new, vel2_new


def animate_particle_movement(positions, velocities, dt, total_time):
    global N_R, N_L, V_R, V_L, S

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1], width_ratios=[2, 1])

    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[1:, 1])

    ax0.set_xlim(0, 1.1)
    ax0.set_ylim(0, 1.1)
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.set_title('Particle Movement')
    ax0.axvline(x=0.5, ymin=0, ymax=1, color='purple', ls='--')

    N = positions.shape[1]
    original_colors = np.where(positions[0, :] >= 0.5, 'red', 'green')
    particles = ax0.scatter(positions[0, :], positions[1, :], color=original_colors)

    text_right = ax0.text(0.8, -0.12, '', transform=ax0.transAxes, ha='center')
    text_left = ax0.text(0.2, -0.12, '', transform=ax0.transAxes, ha='center')

    def update(frame):
        global V_L, V_R, N_R, N_L, S
        nonlocal positions, velocities, particles

        new_time = frame * dt
        positions += velocities * dt

        velocities, n_collision = bound_collision(N, positions, velocities)

        collision_matrix = collisions(positions)
        for i in range(N):
            for j in range(i + 1, N):
                if collision_matrix[i, j]:
                    velocities[:, i], velocities[:, j] = resolve_collision(positions[:, i], velocities[:, i],
                                                                           positions[:, j], velocities[:, j])

        particles.set_offsets(positions.T)

        if frame == 0:
            colors = np.where(positions[0] >= 0.5, 'red', 'green')
            particles.set_color(colors)

        num_right = np.sum(positions[0, :] > 0.5)
        num_left = np.sum(positions[0, :] <= 0.5)
        N_L.append(num_left)
        N_R.append(num_right)
        text_right.set_text('Particles on the right: {}'.format(num_right))
        text_left.set_text('Particles on the left: {}'.format(num_left))

        # avg_velocity_right = np.mean(velocities[0, positions[0, :] > 0.5]**2) if np.sum(positions[0, :] > 0.5) > 0 else 0
        # avg_velocity_left = np.mean(velocities[0, positions[0, :] <= 0.5]**2) if np.sum(positions[0, :] <= 0.5) > 0 else 0
        # V_R.append(avg_velocity_right)
        # V_L.append(avg_velocity_left)
        vel_total = velocities[0, :] ** 2 + velocities[1, :] ** 2

        avg_vel_total_right = np.mean(vel_total[positions[0, :] > 0.5])
        avg_vel_total_left = np.mean(vel_total[positions[0, :] <= 0.5])

        V_R.append(avg_vel_total_right)
        V_L.append(avg_vel_total_left)

        num_right = max(num_right, 1)  # Prevent log(0)
        num_left = max(num_left, 1)  # Prevent log(0)
        S_value = N * np.log(N) - num_right * np.log(num_right) - (N - num_right) * np.log(N - num_right)
        S.append(S_value)

        ax1.clear()
        ax1.plot(N_L, "-", color="green", label="N_L")
        ax1.plot(N_R, "-", color="red", label="N_R")
        ax1.set_xlim(0, total_time / dt)
        ax1.set_ylim(0, N)
        ax1.axhline(y= N / 2, color='purple', linestyle='--')
        ax1.set_title("Number of Particles")
        ax1.legend()

        ax2.clear()
        ax2.plot(V_L, "-", color="green", label="V_L^2")
        ax2.plot(V_R, "-", color="red", label="V_R^2")
        ax2.set_xlim(0, total_time / dt)
        ax2.set_ylim(0, 0.5*np.max(vel_total))
        ax2.set_title("Velocity Squared")
        ax2.legend()

        ax3.clear()
        ax3.plot(S, color="red")
        ax3.set_xlim(0, total_time / dt)
        ax3.set_ylim(0, N * np.log(2) + 5)
        ax3.set_ylabel(r"$\Delta S$/k")
        ax3.axhline(y=N * np.log(2), color='purple', linestyle='--', label=r"N$\log2$")
        ax3.set_title("Entropy")
        ax3.legend()

        return particles, text_right, text_left, ax1, ax2, ax3

    plt.rcParams['animation.embed_limit'] = 100  # Increase the embed limit to 100MB
    ani = FuncAnimation(fig, update, frames=np.arange(0, total_time, dt), blit=False)
    plt.tight_layout()
    plt.close()

    html_str = ani.to_jshtml()
    return html_str


st.title("Particle Simulation")

# Add custom CSS to adjust the layout
st.markdown("""
    <style>
        .main .block-container {
            max-width: 90%;
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

# Sidebar to change simulation parameters
st.sidebar.title("Simulation Parameters")
N = st.sidebar.slider("Number of Particles", min_value=10, max_value=500, value=100, step=10)
left_side_only = st.sidebar.checkbox("One side generator", value=True)
min_pos = 0
max_pos = 0.35 if left_side_only else 1.0
min_speed = -1
max_speed = 1.0
T_L = st.sidebar.slider("Temperature of particles on the left", min_value=0, max_value=500, value=20, step=20)
T_R = st.sidebar.slider("Temperature of particles on the right", min_value=0, max_value=500, value=500, step=20)
dt = st.sidebar.slider("Time step: dt (s)", min_value=0.01, max_value=0.2, value=0.1, step=0.01)
total_time = st.sidebar.slider("Total Simulation Time (s)", min_value=0.1, max_value=5.0, value=5.0, step=0.1)

if st.sidebar.button("Run Simulation"):
    positions = generuj_czastki(N, min_pos, max_pos, rounding=3)
    velocities = generate_velocity_matrix2(N, positions, T_L=T_L, T_R=T_R)

    animation_html = animate_particle_movement(positions, velocities, dt, total_time)
    components.html(animation_html, height=1000, scrolling=False)
