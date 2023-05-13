from time import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from Streamfunctions import *
from MatrixConstructors import *
from ParallelMatrixConstructors import *
from Optimize import *
from ReadWriteArray import *
from PlottingMethods import *


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

    # System matrix generation and saving
    Sx = matrixGenerator(N, M, res, "x", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01]) #AssembleSx(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sy = matrixGenerator(N, M, res, "y", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01]) #AssembleSy(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sz = matrixGenerator(N, M, res, "z", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01]) #AssembleSz(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    try:
        write_array_to_file(Sx, "HarmonicsApproach/Matrices/Sx_N{}_M{}_res{}_Gen_{}.txt".format(N,M,res, matrixGenerator.__name__) )
        write_array_to_file(Sy, "HarmonicsApproach/Matrices/Sy_N{}_M{}_res{}_Gen_{}.txt".format(N,M,res, matrixGenerator.__name__) )
        write_array_to_file(Sz, "HarmonicsApproach/Matrices/Sz_N{}_M{}_res{}_Gen_{}.txt".format(N,M,res, matrixGenerator.__name__) )
    except:
        print("At least one of the filenames was already taken")

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

def TestReadAndOptPCB(N, M, res, matrixGenerator = P_AssembleS):
    """
    Reads the system matrices from files and uses those to optimizes a printed circuit board (PCB) design.

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

    # System matrix generation and saving
    try:
        Sx = read_array_from_file("HarmonicsApproach/Matrices/Sx_N{}_M{}_res{}_Gen_{}.txt".format(N,M,res, matrixGenerator.__name__) )
        Sy = read_array_from_file("HarmonicsApproach/Matrices/Sy_N{}_M{}_res{}_Gen_{}.txt".format(N,M,res, matrixGenerator.__name__) )
        Sz = read_array_from_file("HarmonicsApproach/Matrices/Sz_N{}_M{}_res{}_Gen_{}.txt".format(N,M,res, matrixGenerator.__name__) )
    except:
        print("File does not exist, generate the matrix first")

    Q = Sx.T@Sx + Sy.T@Sy + Sz.T@Sz
    J = np.ones((res**3, res**3))
    R = Sz.T@J@Sz  #Sx.T@J@Sx + Sy.T@J@Sy + Sz.T@J@Sz

    #Initial conditions
    uNorm = 1#N*M
    u_start = (uNorm/N*M)*np.zeros(N*M)
    u_start[0] = 1
    
    #optimization
    func = lambda x: Uniformity(x, Q, R, res )
    jac = lambda x: GradUniformity(x, Q, R, res)
    hess = lambda x: HessUniformity(x, Q, R, res, N*M)
    optres = minimize(func, u_start, jac = jac, hess=hess)
    print(optres) 
    u_opt = optres.x/np.linalg.norm(optres.x)
    V_opt = Uniformity(u_opt, Q, R, res)
    B_opt = [1/res**3*np.sum(Sx@u_opt), 1/res**3*np.sum(Sy@u_opt), 1/res**3*np.sum(Sz@u_opt)]

    #Plotting
    fig3, ax3 = PlotSol(u_opt, N, M, res, V_opt, B_opt)
    fig4, ax4 = plt.subplots(1,1)
    ax4.imshow(u_opt.reshape((N,M)))
    
    return u_opt, V_opt

def TestReadAndOptPCB_Loop(N2, M2, res, matrixGenerator = P_AssembleS, solutionplots = False):
    """
    Reads the system matrices from files and uses those to optimizes a printed circuit board (PCB) design.

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

    # System matrix generation and saving
    try:
        Sx_large = read_array_from_file("HarmonicsApproach/Matrices/Sx_N{}_M{}_res{}_Gen_{}.txt".format(N2,M2,res, matrixGenerator.__name__) )
        Sy_large = read_array_from_file("HarmonicsApproach/Matrices/Sy_N{}_M{}_res{}_Gen_{}.txt".format(N2,M2,res, matrixGenerator.__name__) )
        Sz_large = read_array_from_file("HarmonicsApproach/Matrices/Sz_N{}_M{}_res{}_Gen_{}.txt".format(N2,M2,res, matrixGenerator.__name__) )
    except:
        print("File does not exist, generate the matrix first")

    solutionlist = []
    Vlist = []
    Blist = []
    
    for N1 in range(1,N2+1):
        M1 = N1
        Sx = extract_subarray(Sx_large, N1, M1, N2, M2)
        Sy = extract_subarray(Sy_large, N1, M1, N2, M2)
        Sz = extract_subarray(Sz_large, N1, M1, N2, M2)
        print(Sx.shape)

        Q = Sx.T@Sx + Sy.T@Sy + Sz.T@Sz
        J = np.ones((res**3, res**3))
        R = Sz.T@J@Sz  #Sx.T@J@Sx + Sy.T@J@Sy + Sz.T@J@Sz

        #Initial conditions
        uNorm = 1#N*M
        u_start = (uNorm/N1*M1)*np.zeros(N1*M1)
        u_start[0] = 1
        
        #optimization
        func = lambda x: Uniformity(x, Q, R, res )
        jac = lambda x: GradUniformity(x, Q, R, res)
        hess = lambda x: HessUniformity(x, Q, R, res, N1*M1)
        optres = minimize(func, u_start, jac = jac, hess=hess)
        print(optres) 
        u_opt = optres.x/np.linalg.norm(optres.x)
        solutionlist +=[u_opt]
        V_opt = Uniformity(u_opt, Q, R, res)
        Vlist += [V_opt]
        B_opt = [1/res**3*np.sum(Sx@u_opt), 1/res**3*np.sum(Sy@u_opt), 1/res**3*np.sum(Sz@u_opt)]
        Blist += [B_opt]

        #Plotting
        if solutionplots:
            fig3, ax3 = PlotSol(u_opt, N1, M1, res, V_opt, B_opt)
            fig4, ax4 = plt.subplots(1,1)
            ax4.imshow(u_opt.reshape((N1,M1)))

    fig5, ax5 = plt.subplots(1,1)
    fig6, ax6 = plt.subplots(1,1)
    ax5.plot(range(1,N2+1),Vlist, marker = ".", linestyle = "None")
    ax6.plot(range(1,N2+1),np.array(Blist)[:,2],  marker = ".", linestyle = "None")
    ax5.set_xlabel("Number of harmonics")
    ax5.set_ylabel("Non-uniformity")
    ax5.set_ylim((0, 1.2*max(Vlist)))
    ax6.set_xlabel("Number of harmonics")
    ax6.set_ylabel("Magnetic field in Z direction")

    
    return solutionlist, Vlist, Blist



if __name__ == "__main__":

    res = 18 #Must be even for symmetry!
    N = 16
    M = N
    u, V = TestGenAndOptPCB(N, M, res, matrixGenerator = P_AssembleSSymm)
    #ulist, Vlist, Blist = TestReadAndOptPCB_Loop(N, M, res, matrixGenerator = P_AssembleSSymm)
    #print("Harmonics:", N, " Uniformity: ", V)

    plt.show()
    
