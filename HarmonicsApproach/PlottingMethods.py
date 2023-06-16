import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from Streamfunctions import *

FACTOR = 1e7*1e-3
def PlotSol(u, N, M, Res, V, B, tar_dir, fig = None, ax = None):
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
    plt.style.use("default")
    if tar_dir=="z":
        HarmStreamFunc = HarmStreamFunc_Ztar
    elif tar_dir == "x":
        HarmStreamFunc = HarmStreamFunc_Xtar
    x = np.linspace(-1,1, 1000) 
    y = np.linspace(-1,1, 1000)
    X, Y = np.meshgrid(x, y)
    Sol = np.zeros(X.shape)
    N = np.sqrt(u.size)
    for i, c in enumerate(u):
        m = i//N
        n = i%N
        Sol += c*HarmStreamFunc(X, Y, n, m)
    if fig == None and ax == None:
        fig, ax = plt.subplots(1,1, figsize = (8,6))
    pc = ax.pcolormesh(X, Y, Sol, vmin = - max(np.max(Sol), np.min(Sol)), vmax = max(np.max(Sol), np.min(Sol)))
    ax.set_aspect('equal')
    fig.colorbar(pc)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Current Potential for N=M={}, Res={}, U={:.5f}, B_{}={:.4f} mT".format(2*int(N)+1+(tar_dir=="x"), Res, V, tar_dir, np.linalg.norm(B)/FACTOR))

    #fig2, ax2 = plt.subplots(1,1)
    #pc2 = ax2.contour(X, Y, Sol, levels = 20)
    #ax2.set_aspect('equal')
    #fig2.colorbar(pc2)
    #ax2.set_xlabel("x [m]")
    #ax2.set_ylabel("y [m]")
    #ax2.set_title("Current Loops for N={}, M={}, Res={}, U={:.5f}, B_z={:5f}".format(int(N), M, Res, V, np.linalg.norm(B)))
    #plt.show()
    plt.style.use("seaborn-v0_8-dark")
    return fig, ax


def PlotSolMinimal(u, tar_dir, fig = None, ax = None):
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
    if tar_dir=="z":
        HarmStreamFunc = HarmStreamFunc_Ztar
    elif tar_dir == "x":
        HarmStreamFunc = HarmStreamFunc_Xtar
    x = np.linspace(-1,1, 1000) 
    y = np.linspace(-1,1, 1000)
    X, Y = np.meshgrid(x, y)
    Sol = np.zeros(X.shape)
    N = np.sqrt(u.size)
    for i, c in enumerate(u):
        m = i//N
        n = i%N
        Sol += c*HarmStreamFunc(X, Y, n, m)
    if fig == None and ax == None:
        fig, ax = plt.subplots(1,1)
    print(ax)
    pc = ax.pcolormesh(X, Y, Sol, vmin = - max(np.max(Sol), np.min(Sol)), vmax = max(np.max(Sol), np.min(Sol)))
    ax.set_aspect('equal')
    fig.colorbar(pc)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    #ax.set_title("Current Potential for N={}, M={}, Res={}, U={:.5f}, B_{}={:5f}".format(int(N), M, Res, V, tar_dir, np.linalg.norm(B)))

    #fig2, ax2 = plt.subplots(1,1)
    #pc2 = ax2.contour(X, Y, Sol, levels = 20)
    #ax2.set_aspect('equal')
    #fig2.colorbar(pc2)
    #ax2.set_xlabel("x [m]")
    #ax2.set_ylabel("y [m]")
    #ax2.set_title("Current Loops for N={}, M={}, Res={}, U={:.5f}, B_z={:5f}".format(int(N), M, Res, V, np.linalg.norm(B)))
    #plt.show()
    return fig, ax


def PlotSol3D(u, Sx, Sy, Sz, tar_dir, N, M, resolution):
    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    if tar_dir=="z":
        HarmStreamFunc = HarmStreamFunc_Ztar
    elif tar_dir == "x":
        HarmStreamFunc = HarmStreamFunc_Xtar

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
    ax.plot_surface(X, Y, np.zeros_like(Sol), facecolors=plt.cm.viridis((Sol -np.min(Sol))/(np.max(Sol)-np.min(Sol))), shade=False, antialiased=False, zorder=-1)

    V = [-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]

    vertices = [
        [V[0], V[2], V[4]],
        [V[1], V[2], V[4]],
        [V[1], V[3], V[4]],
        [V[0], V[3], V[4]],
        [V[0], V[2], V[5]],
        [V[1], V[2], V[5]],
        [V[1], V[3], V[5]],
        [V[0], V[3], V[5]]
    ]

    edges = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[3]],
        [vertices[3], vertices[0]],
        [vertices[4], vertices[5]],
        [vertices[5], vertices[6]],
        [vertices[6], vertices[7]],
        [vertices[7], vertices[4]],
        [vertices[0], vertices[4]],
        [vertices[1], vertices[5]],
        [vertices[2], vertices[6]],
        [vertices[3], vertices[7]]
    ]
    ax.set_zlim(-0.5,1.5)
    ax.set_aspect("equal")

    edge_collection = Poly3DCollection(edges, linewidths=2.5, edgecolors='red')
    ax.add_collection(edge_collection)
    #ax.set_xlim(-1.25,1.25)
    #ax.set_ylim(-1.25,1.25)

    NM = N * M
    x2 = np.linspace(V[0], V[1], resolution)  # Discretize the target volume into a grid
    y2 = np.linspace(V[2], V[3], resolution)
    z2 = np.linspace(V[4], V[5], resolution)
    Z2, Y2, X2 = np.meshgrid(z2, y2, x2, indexing='ij')
    X2 = X2.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
    Y2 = Y2.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
    Z2 = Z2.reshape((resolution ** 3), order="C")  # Order the volume lexicographically

    Bx = Sx@u
    By = Sy@u
    Bz = Sz@u
    B_max = max(np.linalg.norm(np.array([Bx, By, Bz]), axis = 1))
    length = 10/resolution


    ax.quiver(X2, Y2, Z2, length*Bx/B_max, length*By/B_max, length*Bz/B_max, normalize=False, color = "k")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    #ax.view_init(elev=48, azim=-65)
    ax.view_init(elev=0, azim=-90)
    
    return fig, ax
