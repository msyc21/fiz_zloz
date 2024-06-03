import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def pos_init(N):
    #pos = np.zeros((2, N), dtype=int)  #zaczynamy od pzycji 0,0 dla kazdej czastki
    posx = np.random.randint(5,size=N)
    posy = np.random.randint(10, size=N)
    pos = np.array([posx,posy])
    return pos

def dir_init(N):
    dir = np.random.choice(4, N)  # zakladajac kierunki up down left right 0,1,2,3
    return dir

def update_positions(pos, dir, grid_size):
    for i in range(pos.shape[1]):
        dir[i] = np.random.choice(4) #random kierunek w kazdym kroku

        if dir[i] == 0:  # gora
            pos[1, i] += 1
        elif dir[i] == 1:  # dol
            pos[1, i] -= 1
        elif dir[i] == 2:  #lewo
            pos[0, i] -= 1
        elif dir[i] == 3:  # prawo
            pos[0, i] += 1

        #odbijanie od scian
        if pos[0, i] < 0:
            pos[0, i] = 0
        elif pos[0, i] >= grid_size:
            pos[0, i] = grid_size - 1

        if pos[1, i] < 0:
            pos[1, i] = 0
        elif pos[1, i] >= grid_size:
            pos[1, i] = grid_size - 1
    #print("pos:",pos)

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


        particles.set_offsets(pos.T) #aktualizacja pozycji czastek

        return particles,

    ani = FuncAnimation(fig, update, frames=np.arange(0, total_steps, dt), blit=True, repeat=False)

    plt.close()

    return HTML(ani.to_jshtml())

N = 10
grid_size = 10  # rozmiar siatki
total_steps = 100  # ilosc krokow w symulacji
dt = 1


poss = pos_init(N)
dirs = dir_init(N)

HTML_output = animate_particle_movement(poss, dirs, grid_size, total_steps, dt)
display(HTML_output)
