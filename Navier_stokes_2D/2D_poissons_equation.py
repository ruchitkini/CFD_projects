import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

""" Define the parameters """
nx = 50     # number of grid points in x-direction
ny = 50     # number of grid points in y-direction
nt = 100   # time steps
xmin = 0
xmax = 1
ymin = 0
ymax = 1

dx = (xmax - xmin) / (nx - 1)   # grid spacing in x-direction
dy = (ymax - ymin) / (ny - 1)   # grid spacing in y-direction

# Initialization
p = np.zeros((ny, nx))
pd = np.zeros((ny, nx))
b = np.zeros((ny, nx))

x = np.linspace(xmin, xmax, nx)
y = np.linspace(xmin, xmax, ny)

# Source
b[int(ny / 4), int(nx / 4)] = 100
b[int(3 * ny / 4), int(3 * nx / 4)] = -100


for it in range(nt):

    pd = p.copy()

    p[1:-1,1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy**2 + (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx**2 - b[1:-1, 1:-1] * dx**2 * dy**2) / (2 * (dx**2 + dy**2)))
    print()
    # p[1:-1,1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy**2 + (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx**2 - b[1:-1, 1:-1] * dx**2 * dy**2) / (2 * (dx**2 + dy**2)))


    p[0, :] = 0
    p[ny-1, :] = 0
    p[:, 0] = 0
    p[:, nx-1] = 0



def plot2D(x, y, p):
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)
    ax.view_init(30, 225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

plot2D(x, y, p)

