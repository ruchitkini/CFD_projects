""" This is starting a simple code - Heat equation in 1D """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L = 0.1             # wall thickness
n = 20              # number of grid points
T0 = 0              # initial condition
T1s, T2s = 40, 20   # boundary condition
dx = L / n          # grid spacing
alpha = 0.0001      # thermal diffusivity
t_final = 60        # final time
dt = 0.1            # time step

# Creating grid in space and time domain
x = np.linspace(dx / 2, L - dx / 2, n)
T = np.ones(n) * T0
dT_dt = np.empty(n)
t = np.arange(0, t_final+dt, dt)

# Setting up the figure and axis
fig, ax = plt.subplots()
line, = ax.plot(x, T)
ax.set_xlim(0, L)
ax.set_ylim(0, 50)
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Temperature (C)')
title = ax.set_title('')

def update(frame):
    global T
    # Update temperature distribution
    for i in range(1, n - 1):
        dT_dt[i] = alpha * (T[i + 1] - 2 * T[i] + T[i - 1]) / dx ** 2
    dT_dt[0] = alpha * (T[1] - 2 * T[0] + T1s) / dx ** 2
    dT_dt[n - 1] = alpha * (T2s - 2 * T[n - 1] + T[n - 2]) / dx ** 2
    T = T + dT_dt * dt

    line.set_ydata(T)
    title.set_text(f'Time = {frame * dt:.2f}s')
    return line, title

# Creating the animation
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=10, blit=False, repeat=False)

# Display the animation
plt.show()
