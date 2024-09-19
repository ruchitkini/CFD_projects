import numpy as np
import matplotlib.pyplot as plt

""" Define the parameters """
lx, ly = 1, 1   # length of cavity in x and y direction respectively
nx, ny = 41, 41     # number of grid points in x and y direction respectively
nt = 500
nit = 50
c = 1
dx = lx / (nx - 1)      # grid spacing in x-direction
dy = ly / (ny - 1)      # grid spacing in y-direction

x = np.linspace(0, lx, nx) 
y = np.linspace(0, ly, ny)
X,Y = np.meshgrid(x, y)     # create the meshgrid

rho = 1     # density of fluid
nu = 0.1    # kinematic viscosity of fluid
dt = 0.001  # time step

u = np.zeros((nx, ny))      # velocity in x-direction
v = np.zeros((nx, ny))      # velocity in y-direction
p = np.zeros((nx, ny))      # pressure
b = np.zeros((nx, ny))








plt.scatter(X, Y)
plt.show()

