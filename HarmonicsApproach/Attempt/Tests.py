from time import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize

from Streamfunctions import *
from MatrixConstructors import *
from Optimize import *


def Test_Conv():
    print(psiz_conv_dely_harm(0.5, 0.5, 0.5, 0, 0,))
    print(psiz_conv_delx_harm(1.0, 1.5, 2.5, 0, 0,))
    print(psixy_conv_delxy_harm(0.5, 0.5, 0.5, 0, 0))
    return

def PlotSol(u, N, M, Res, V):
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
    ax.set_title("Current Potential for N = {}, M= {}, Res = {}, U = {:.3f}".format(int(N), M, Res, V))

    fig2, ax2 = plt.subplots(1,1)
    pc2 = ax2.contour(X, Y, Sol, levels = 20)
    ax2.set_aspect('equal')
    fig2.colorbar(pc2)
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Current Loops for N = {}, M= {}, Res = {}, U = {:.3f}".format(int(N), M, Res, V))
    #plt.show()
    return fig, ax

def PlotSol3D(u, Sx, Sy, Sz):
    return


def TestFullVolume(N, M, res):
    uNorm = 1#N*M
    u_start = (uNorm/N*M)*np.zeros(N*M)
    u_start[0] = 1
    #u_start[1] = N*M/np.sqrt(3)
    #u_start[N] = N*M/np.sqrt(3)
    #fig1,  ax1 = PlotSol(u_start)
    Sx = AssembleSx(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sy = AssembleSy(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sz = AssembleSz(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Q = Sx.T@Sx + Sy.T@Sy + Sz.T@Sz
    J = np.ones((res**3, res**3))
    R = Sz.T@J@Sz  #Sx.T@J@Sx + Sy.T@J@Sy + Sz.T@J@Sz
    #print(Q, R, Sx)
    #print(null_space(Q).shape)
    #plt.imshow(AssembleSx(5, 5, 8), aspect='auto') # time is O(N*M*res**3)
    #plt.imshow(AssembleSy(5, 5, 8), aspect='auto') # time is O(N*M*res**3)
    #plt.imshow(AssembleSz(5, 5, 8), aspect='auto') # time is O(N*M*res**3)
    #u, Vs, u_best, V_best = line_search_line_min(Q, R, res, uNorm, u_start, 500, alpha = 0.0)
    #print(u)
    func = lambda x: Uniformity(x, Q, R, res )
    jac = lambda x: GradUniformity(x, Q, R, res)
    optres = minimize(func, u_start, jac = jac)
    print(optres) 
    u, V, u_best, V_best = line_search_line_min(Q, R, res, uNorm, u_start, 500, alpha = 0.0)
    #print(u_best)
    u_opt = optres.x/np.linalg.norm(optres.x)
    V_opt = Uniformity(u_opt, Q, R, res)
    print(u_best-u_opt, np.linalg.norm(u_best-u_opt))
    #fig2, ax2 = PlotSol(u)
    fig3, ax3 = PlotSol(u_opt, N, M, res, V_opt)
    fig4, ax4 = plt.subplots(1,1)
    ax4.imshow(u_opt.reshape((N,M)))
    
    return u_opt, V_opt

def TestFullVolumeEdge(N, M, res):
    uNorm = 1#N*M
    u_start = (uNorm/N*M)*np.zeros(N*M)
    u_start[0] = N*M/np.sqrt(3)
    #u_start[1] = N*M/np.sqrt(3)
    #u_start[N] = N*M/np.sqrt(3)
    #fig1,  ax1 = PlotSol(u_start)
    Sx = AssembleSxSurf(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sy = AssembleSySurf(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sz = AssembleSzSurf(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Q = Sx.T@Sx + Sy.T@Sy + Sz.T@Sz
    J = np.ones((res**3-(res-2)**3, res**3-(res-2)**3))
    R = Sz.T@J@Sz #Sx.T@J@Sx + Sy.T@J@Sy + Sz.T@J@Sz
    #plt.imshow(AssembleSx(5, 5, 8), aspect='auto') # time is O(N*M*res**3)
    #plt.imshow(AssembleSy(5, 5, 8), aspect='auto') # time is O(N*M*res**3)
    #plt.imshow(AssembleSz(5, 5, 8), aspect='auto') # time is O(N*M*res**3)
    u, Vs, u_best, V_best = line_search_line_min(Q, R, res, uNorm, u_start, 5000, alpha = 0.0)
    print(u)
    print(u_best)
    #fig2, ax2 = PlotSol(u)
    fig3, ax3 = PlotSol(u_best, N, M, res, V_best)
    fig4, ax4 = plt.subplots(1,1)
    ax4.imshow(u_best.reshape((N,M)))
    
    return 

def TestFullVolumeEdgeSym(N, M, res):
    uNorm = 1#N*M
    u_start = (uNorm/N*M)*np.zeros(N*M)
    u_start[0] = N*M/np.sqrt(3)
    #u_start[1] = N*M/np.sqrt(3)
    #u_start[N] = N*M/np.sqrt(3)
    #fig1,  ax1 = PlotSol(u_start)
    Sx = AssembleSxSurfZSym(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sy = AssembleSySurfZSym(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sz = AssembleSzSurfZSym(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Q = Sx.T@Sx + Sy.T@Sy + Sz.T@Sz
    J = np.ones((res**3-(res-2)**3, res**3-(res-2)**3))
    R = Sz.T@J@Sz #Sx.T@J@Sx + Sy.T@J@Sy + Sz.T@J@Sz
    #plt.imshow(AssembleSx(5, 5, 8), aspect='auto') # time is O(N*M*res**3)
    #plt.imshow(AssembleSy(5, 5, 8), aspect='auto') # time is O(N*M*res**3)
    #plt.imshow(AssembleSz(5, 5, 8), aspect='auto') # time is O(N*M*res**3)
    u, Vs, u_best, V_best = line_search_line_min(Q, R, res, uNorm, u_start, 5000, alpha = 0.0)
    print(u)
    print(u_best)
    #fig2, ax2 = PlotSol(u)
    fig3, ax3 = PlotSol(u_best, N, M, res, V_best)
    fig4, ax4 = plt.subplots(1,1)
    ax4.imshow(u_best.reshape((N,M)))
    
    return



if __name__ == "__main__":
    u_list = []
    V_list = []
    for N in range(10,11):
        #P   = 4         #power [W]
        #rho = 1         #resistivity [?]
        #Dz  = 0.0001    #trace thickness [m]
        res = 10 #Must be even!
        N = N
        M = N
        u, V = TestFullVolume(N, M, res)
        #TestFullVolumeEdge(N, M, res)
        #TestFullVolumeEdgeSym(N, M, res)
        print(N, V)
        u_list +=[u]
        V_list +=[V]
    print(u_list)
    print(V_list)

    plt.show()
    
