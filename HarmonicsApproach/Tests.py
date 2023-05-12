from time import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize

from Streamfunctions import *
from MatrixConstructors import *
from ParallelMatrixConstructors import *
from Optimize import *

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
    ax.set_title("Current Potential for N={}, M={}, Res={}, U={:.4f}, B_avg={:4f}".format(int(N), M, Res, V, np.linalg.norm(B)))

    fig2, ax2 = plt.subplots(1,1)
    pc2 = ax2.contour(X, Y, Sol, levels = 20)
    ax2.set_aspect('equal')
    fig2.colorbar(pc2)
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Current Loops for N={}, M={}, Res={}, U={:.4f}, B_avg={:4f}".format(int(N), M, Res, V, np.linalg.norm(B)))
    #plt.show()
    return fig, ax

def PlotSol3D(u, Sx, Sy, Sz):
    return


def TestGenAndOptPCB(N, M, res, matrixGenerator = P_AssembleS):
    """
    Generates the system matrices and uses those to optimizes a printed circuit board (PCB) design.

    Args:
        N (int): Number of harmonics in the x-direction.
        M (int): Number of harmonics in the y-direction.
        res (int): Number of grid points in the target volume in each dimension (x, y, z).
        matrixGenerator (function, optional): Function to generate a system matrix S for magnetic fields in the x, y or z direction
                                              Defaults to P_AssembleS.

    Returns:
        numpy.ndarray: Optimal solution u_opt.
        float: Value of the objective function V_opt.

    """
    #Initial conditions
    uNorm = 1#N*M
    u_start = (uNorm/N*M)*np.zeros(N*M)
    u_start[0] = 1

    # matrix generation
    Sx = matrixGenerator(N, M, res, "x", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01]) #AssembleSx(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sy = matrixGenerator(N, M, res, "y", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01]) #AssembleSy(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sz = matrixGenerator(N, M, res, "z", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01]) #AssembleSz(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Q = Sx.T@Sx + Sy.T@Sy + Sz.T@Sz
    J = np.ones((res**3, res**3))
    R = Sz.T@J@Sz  #Sx.T@J@Sx + Sy.T@J@Sy + Sz.T@J@Sz
    
    #optimization
    func = lambda x: Uniformity(x, Q, R, res )
    jac = lambda x: GradUniformity(x, Q, R, res)
    hess = lambda x: HessUniformity(x, Q, R, res, N*M)
    optres = minimize(func, u_start, jac = jac, hess=hess)
    print(optres) 
    u_opt = optres.x/np.linalg.norm(optres.x)
    V_opt = Uniformity(u_opt, Q, R, res)
    B_opt = [1/res**3*np.sum(Sx@u_opt), 1/res**3*np.sum(Sy@u_opt), 1/res**3*np.sum(Sz@u_opt)]
    #u, V, u_best, V_best = line_search_line_min(Q, R, res, uNorm, u_start, 500, alpha = 0.0)
    #print(u_best)

    #Plotting
    fig3, ax3 = PlotSol(u_opt, N, M, res, V_opt, B_opt)
    fig4, ax4 = plt.subplots(1,1)
    ax4.imshow(u_opt.reshape((N,M)))
    
    return u_opt, V_opt



if __name__ == "__main__":
    u_list = []
    V_list = []
    for N in range(5,12):
        #P   = 4         #power [W]
        #rho = 1         #resistivity [?]
        #Dz  = 0.0001    #trace thickness [m]
        res = 12 #Must be even for symmetry!
        N = N
        M = N
        u, V = TestGenAndOptPCB(N, M, res, matrixGenerator = P_AssembleSSymm)
        #TestFullVolumeEdge(N, M, res)
        #TestFullVolumeEdgeSym(N, M, res)
        print(N, V)
        u_list +=[u]
        V_list +=[V]
    print(u_list)
    print(V_list)

    plt.show()
    
