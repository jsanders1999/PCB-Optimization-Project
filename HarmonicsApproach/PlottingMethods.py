import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from Streamfunctions import *


def PlotSol(u, N, M, Res, V, B):
    """
    Plots the current potential and current loops for a PCB design.

    Args:
        u (numpy.ndarray): Solution vector.
        N (int): Number of harmonics in the x-direction.
        M (int): Number of harmonics in the y-direction.
        res (int): Number of grid points in the target volume in each dimension (x, y, z).
        V (float): Value of the objective function, the uniformity, which is the normalized variance of the magnetic field in the target volume.
        B (list): List of average magnetic field components [Bx, By, Bz].

    Returns:
        matplotlib.figure.Figure: Figure object.
        matplotlib.axes.Axes: Axes object.

    """
    x = np.linspace(-1,1, 1000) 
    y = np.linspace(-1,1, 1000)
    X, Y = np.meshgrid(x, y)
    Sol = np.zeros(X.shape)
    N = np.sqrt(u.size)
    for i, c in enumerate(u):
        m = i//N
        n = i%N
        Sol += c*HarmStreamFunc(X, Y, n, m)

    fig, ax = plt.subplots(1,1)
    pc = ax.pcolormesh(X, Y, Sol)
    ax.set_aspect('equal')
    fig.colorbar(pc)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Current Potential for N={}, M={}, Res={}, U={:.5f}, B_z={:5f}".format(int(N), M, Res, V, np.linalg.norm(B)))

    fig2, ax2 = plt.subplots(1,1)
    pc2 = ax2.contour(X, Y, Sol, levels = 20)
    ax2.set_aspect('equal')
    fig2.colorbar(pc2)
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Current Loops for N={}, M={}, Res={}, U={:.5f}, B_z={:5f}".format(int(N), M, Res, V, np.linalg.norm(B)))
    #plt.show()
    return fig, ax

def PlotSol3D(u, Sx, Sy, Sz):
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-1,1, 1000) 
    y = np.linspace(-1,1, 1000)
    X, Y = np.meshgrid(x, y)
    Sol = np.zeros(X.shape)
    N = np.sqrt(u.size)
    for i, c in enumerate(u):
        m = i//N
        n = i%N
        Sol += c*HarmStreamFunc(X, Y, n, m)

    # Plot the flat plane on the XY plane
    ax.plot_surface(X, Y, np.zeros_like(Z), facecolors=plt.cm.viridis(Z), shade=False)

    # Plot cube edges
    V = [-0.5, 0.5, -0.5, 0.5, 0.01, 1.01]
    x_v = [V[0], V[1], V[1], V[0], V[0], V[1], V[1], V[0]]
    y_v = [V[2], V[2], V[3], V[3], V[2], V[2], V[3], V[3]]
    z_v = [V[4], V[4], V[4], V[4], V[5], V[5], V[5], V[5]]
    ax.plot(x_v, y_v, z_v, color='black')

    return