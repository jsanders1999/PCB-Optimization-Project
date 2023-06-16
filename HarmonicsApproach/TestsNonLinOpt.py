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
from Contour import *


def TestGenAndOptPCB(N, M, res, tar_dir,  matrixGenerator = P_AssembleS):
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
    if tar_dir != "x" and tar_dir != "z":
        raise ValueError("Target direction must be 'x' or 'z'")
    
    #Initial conditions
    uNorm = 1#N*M
    u_start = (uNorm/N*M)*np.zeros(N*M)
    u_start[0] = 1

    # System matrix generation and saving
    Sx = matrixGenerator(N, M, res, "x", tar_dir) #AssembleSx(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sy = matrixGenerator(N, M, res, "y", tar_dir) #AssembleSy(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sz = matrixGenerator(N, M, res, "z", tar_dir) #AssembleSz(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    try:
        write_array_to_file(Sx, "HarmonicsApproach/Matrices/Sx_{}_N{}_M{}_res{}_Gen_{}.txt".format(tar_dir, N, M, res, matrixGenerator.__name__) )
        write_array_to_file(Sy, "HarmonicsApproach/Matrices/Sy_{}_N{}_M{}_res{}_Gen_{}.txt".format(tar_dir, N, M, res, matrixGenerator.__name__) )
        write_array_to_file(Sz, "HarmonicsApproach/Matrices/Sz_{}_N{}_M{}_res{}_Gen_{}.txt".format(tar_dir, N, M, res, matrixGenerator.__name__) )
    except:
        print("At least one of the filenames was already taken")

    Q = Sx.T@Sx + Sy.T@Sy + Sz.T@Sz
    J = np.ones((res**3, res**3))
    if tar_dir == "z":
        R = Sz.T@J@Sz  #Sx.T@J@Sx + Sy.T@J@Sy + Sz.T@J@Sz
    elif tar_dir == "x":
        R = Sx.T@J@Sx 
    
    #optimization
    C = 1e-5 #power constraint coefficient
    func = lambda x: Uniformity(x, Q, R, res ) + C*(x.T@x-1)**2
    jac = lambda x: GradUniformity(x, Q, R, res) + 4*C*(x.T@x-1)*x
    hess = lambda x: HessUniformity(x, Q, R, res, N*M)
    optres = minimize(func, u_start, jac = jac, hess=hess)
    print(optres) 
    u_opt = optres.x/np.linalg.norm(optres.x)
    V_opt = Uniformity(u_opt, Q, R, res)
    B_opt = [1/res**3*np.sum(Sx@u_opt), 1/res**3*np.sum(Sy@u_opt), 1/res**3*np.sum(Sz@u_opt)]
    #u, V, u_best, V_best = line_search_line_min(Q, R, res, uNorm, u_start, 500, alpha = 0.0)
    #print(u_best)

    #Plotting
    fig3, ax3 = PlotSol(u_opt, N, M, res, V_opt, B_opt, tar_dir)
    fig4, ax4 = plt.subplots(1,1)
    ax4.imshow(u_opt.reshape((N,M)))
    
    return u_opt, V_opt




def TestGenAndOptDoublePCB(N, M, res, matrixGenerator = P_AssembleS):
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
    u_start = (uNorm/(2*N*M))*np.zeros((2*N*M))
    u_start[0] = 1
    u_start[N*M] = 1

    # System matrix generation and saving
    z_offset = 0.5
    Sx1 = matrixGenerator(N, M, res, "x", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  z_offset, 1+z_offset]) #AssembleSx(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sy1 = matrixGenerator(N, M, res, "y", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  z_offset, 1+z_offset]) #AssembleSy(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sz1 = matrixGenerator(N, M, res, "z", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  z_offset, 1+z_offset]) #AssembleSz(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sx2 = matrixGenerator(N, M, res, "x", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  -1-z_offset, -z_offset]) #AssembleSx(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sy2 = matrixGenerator(N, M, res, "y", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  -1-z_offset, -z_offset]) #AssembleSy(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sz2 = matrixGenerator(N, M, res, "z", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  -1-z_offset, -z_offset]) #AssembleSz(N, M, res, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.01, 1.01])
    Sx = np.hstack((Sx1, Sx2))
    print(Sx.shape)
    Sy = np.hstack((Sy1, Sy2))
    Sz = np.hstack((Sz1, Sz2))
    #try:
    #    write_array_to_file(Sx, "HarmonicsApproach/Matrices/DoublePCB_Ztar_Sx_N{}_M{}_res{}_Gen_{}.txt".format(N,M,res, matrixGenerator.__name__) )
    #    write_array_to_file(Sy, "HarmonicsApproach/Matrices/DoublePCB_Ztar_Sz_N{}_M{}_res{}_Gen_{}.txt".format(N,M,res, matrixGenerator.__name__) )
    #    write_array_to_file(Sz, "HarmonicsApproach/Matrices/DoublePCB_Ztar_Sz_N{}_M{}_res{}_Gen_{}.txt".format(N,M,res, matrixGenerator.__name__) )
    #except:
    #    print("At least one of the filenames was already taken")

    Q = Sx.T@Sx + Sy.T@Sy + Sz.T@Sz
    J = np.ones((res**3, res**3))
    R = Sz.T@J@Sz  #Sx.T@J@Sx + Sy.T@J@Sy + Sz.T@J@Sz
    print(Q.shape, R.shape, u_start.shape)
    
    #optimization
    C = 1e-5
    func = lambda x: Uniformity(x, Q, R, res ) + C*(x.T@x-1)**2
    jac = lambda x: GradUniformity(x, Q, R, res) + 4*C*(x.T@x-1)*x
    constraints = {"type": "eq",
                    "fun": lambda x: (x.T@x-1)**2 ,
                    "jac": lambda x: 4*(x.T@x-1)*x}
    #hess = lambda x: HessUniformity(x, Q, R, res, N1*M1)
    optres = minimize(func, u_start, jac = jac, method = "BFGS", tol = 1e-5)
    print(optres)
    u_opt = optres.x/np.linalg.norm(optres.x)
    print("")
    print(N,M)
    print(np.linalg.norm(optres.x))
    print("jac:", np.linalg.norm(optres.jac))
    print("jac:", np.linalg.norm(jac(optres.x)))
    print("jac:", np.linalg.norm(jac(u_opt)))
    V_opt = Uniformity(u_opt, Q, R, res)
    B_opt = [1/res**3*np.sum(Sx@u_opt), 1/res**3*np.sum(Sy@u_opt), 1/res**3*np.sum(Sz@u_opt)]
    #u, V, u_best, V_best = line_search_line_min(Q, R, res, uNorm, u_start, 500, alpha = 0.0)
    #print(u_best)

    #Plotting
    fig3, ax3 = PlotSol(u_opt[0:N*M], N, M, res, V_opt, B_opt)
    fig4, ax4 = PlotSol(u_opt[N*M:], N, M, res, V_opt, B_opt)
    fig5, ax5 = plt.subplots(1,1)
    ax5.imshow(u_opt[:N*M].reshape((N,M)))
    fig6, ax6 = plt.subplots(1,1)
    ax6.imshow(u_opt[N*M:].reshape((N,M)))
    
    return u_opt, V_opt




def TestReadAndOptPCB(N, M, res, tar_dir, matrixGenerator = P_AssembleS):
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
        Sx = read_array_from_file( "HarmonicsApproach/Matrices/Sx_{}_N{}_M{}_res{}_Gen_{}.txt".format(tar_dir, N, M, res, matrixGenerator.__name__) )
        Sy = read_array_from_file( "HarmonicsApproach/Matrices/Sy_{}_N{}_M{}_res{}_Gen_{}.txt".format(tar_dir, N, M, res, matrixGenerator.__name__) )
        Sz = read_array_from_file( "HarmonicsApproach/Matrices/Sz_{}_N{}_M{}_res{}_Gen_{}.txt".format(tar_dir, N, M, res, matrixGenerator.__name__) )
    except:
        print("File does not exist, generate the matrix first")


    Q = Sx.T@Sx + Sy.T@Sy + Sz.T@Sz
    J = np.ones((res**3, res**3))
    if tar_dir == "z":
        R = Sz.T@J@Sz  #Sx.T@J@Sx + Sy.T@J@Sy + Sz.T@J@Sz
    elif tar_dir == "x":
        R = Sx.T@J@Sx 

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
    fig3, ax3 = PlotSol(u_opt, N, M, res, V_opt, B_opt, tar_dir)
    fig4, ax4 = plt.subplots(1,1)
    ax4.imshow(u_opt.reshape((N,M)))
    
    return u_opt, V_opt

def TestReadAndOptPCB_Loop(N2, M2, res, tar_dir, matrixGenerator = P_AssembleS, solutionplots = False):
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
        Sx_large = read_array_from_file( "HarmonicsApproach/Matrices/Sx_{}_N{}_M{}_res{}_Gen_{}.txt".format(tar_dir, N, M, res, matrixGenerator.__name__) )
        Sy_large = read_array_from_file( "HarmonicsApproach/Matrices/Sy_{}_N{}_M{}_res{}_Gen_{}.txt".format(tar_dir, N, M, res, matrixGenerator.__name__) )
        Sz_large = read_array_from_file( "HarmonicsApproach/Matrices/Sz_{}_N{}_M{}_res{}_Gen_{}.txt".format(tar_dir, N, M, res, matrixGenerator.__name__) )
    except:
        print("File does not exist, generate the matrix first")

    solutionlist = []
    Vlist = []
    Blist = []
    #Initial conditions
    uNorm = 1#N*M
    u_start = np.zeros(1)
    u_start[0] = 1
    for N1 in range(1,N2+1):
        M1 = N1
        Sx = extract_subarray(Sx_large, N1, M1, N2, M2)
        Sy = extract_subarray(Sy_large, N1, M1, N2, M2)
        Sz = extract_subarray(Sz_large, N1, M1, N2, M2)
        #print("SY:", np.linalg.norm(Sy- TurnSxintoSy(Sx, N1, M1)))
        #print(np.where(Sy - TurnSxintoSy(Sx, N1, M1)>1e-5))
        #print(TurnSxintoSy(Sx, N1, M1))
        #print(Sx.shape)
        

        Q = Sx.T@Sx + Sy.T@Sy + Sz.T@Sz
        J = np.ones((res**3, res**3))
        if tar_dir == "z":
            R = Sz.T@J@Sz  #Sx.T@J@Sx + Sy.T@J@Sy + Sz.T@J@Sz
            ind = 2
        elif tar_dir == "x":
            R = Sx.T@J@Sx 
            ind = 0


        #optimization
        C = 1e-6
        func = lambda x: Uniformity(x, Q, R, res ) + C*(x.T@x-1)**2
        jac = lambda x: GradUniformity(x, Q, R, res) + 4*C*(x.T@x-1)*x
        constraints = {"type": "eq",
                       "fun": lambda x: (x.T@x-1)**2 ,
                       "jac": lambda x: 4*(x.T@x-1)*x}
        #hess = lambda x: HessUniformity(x, Q, R, res, N1*M1)
        optres = minimize(func, u_start, jac = jac, method = "BFGS", tol = 1e-5)
        print(optres)
        u_opt = optres.x/np.linalg.norm(optres.x)
        print("")
        print(N1,M1)
        print(np.linalg.norm(optres.x))
        print("jac:", np.linalg.norm(optres.jac))
        print("jac:", np.linalg.norm(jac(optres.x)))
        print("jac:", np.linalg.norm(jac(u_opt)))
        #u_opt, V_opt = bfgs_optimization(func, jac, u_start, max_iter=100000, tol=1e-5)
        #u_opt = u_opt/np.linalg.norm(u_opt)
        #print(u_opt- u_best)
        #print(np.linalg.norm(u_best-u_opt))
        #print(V_opt)
        solutionlist +=[u_opt]
        V_opt = Uniformity(u_opt, Q, R, res)
        Vlist += [V_opt]
        B_opt = [1/res**3*np.sum(Sx@u_opt), 1/res**3*np.sum(Sy@u_opt), 1/res**3*np.sum(Sz@u_opt)]
        Blist += [B_opt]

        u_ls = sp.linalg.solve(Q, Sz.T@np.ones((res**3)))
        u_ls = u_ls/np.linalg.norm(u_ls)
        print(u_ls-u_opt)

        #Plotting
        if solutionplots and N1==N2:
            fig3, ax3 = PlotSol(u_opt, N1, M1, res, V_opt, B_opt, tar_dir)
            fig3, ax3 = PlotSol(u_ls, N1, M1, res, V_opt, B_opt, tar_dir)
            fig3, ax3 = PlotSol3D(u_opt, Sx, Sy, Sz, tar_dir, N, M, res)
            #fig4, ax4 = plt.subplots(1,1)
            #ax4.imshow(u_opt.reshape((N1,M1)))
            #PlotSol3D(u_opt, Sx, Sy, Sz)
            contours = CalcContoursHarmonics(u_opt, tar_dir, N, M)
            CalcUniformityContours(contours, res)
            
        
        
        u_start = np.zeros((N1+1, M1+1))
        u_start[:N1,:M1] = u_opt.reshape((N1,M1))
        u_start = u_start.reshape(((N1+1)*( M1+1)))


    fig5, ax5= plt.subplots(2,1)
    ax5[0].plot(range(1,N2+1),Vlist, marker = ".", linestyle = "None")
    ax5[1].plot(range(1,N2+1),np.array(Blist)[:,ind],  marker = ".", linestyle = "None")
    #ax5[0].set_xlabel("Number of harmonics")
    ax5[0].set_ylabel("Non-uniformity")
    ax5[0].set_yscale("log")
    #ax5.set_ylim((0, 1.2*max(Vlist)))
    ax5[1].set_xlabel("Number of harmonics")
    ax5[1].set_ylabel("Magnetic field in {} direction".format(tar_dir))
    ax5[1].set_yscale("log")

    
    return solutionlist, Vlist, Blist

def CalcContoursHarmonics(u, tar_dir, N, M, num_levels = 30):
    if tar_dir=="z":
        HarmStreamFunc = HarmStreamFunc_Ztar
    elif tar_dir == "x":
        HarmStreamFunc = HarmStreamFunc_Xtar
    x = np.linspace(-1,1, 1000) 
    y = np.linspace(-1,1, 1000)
    X, Y = np.meshgrid(x, y)
    Sol = np.zeros(X.shape)
    for i, c in enumerate(u):
        m = i//N
        n = i%N
        Sol += c*HarmStreamFunc(X, Y, n, m)
    min_spacing = 0.00005
    SolGrad = np.array(np.gradient(Sol))
    print(SolGrad.shape)
    dx = x[1]-x[0]
    max_slope = np.max(np.linalg.norm(SolGrad, axis = 0))
    num_levels = int((np.max(Sol)-np.min(Sol))/(max_slope/dx*min_spacing)) #(np.max(Sol)-np.min(Sol))/min_spacing*SolGrad
    print(num_levels)
    print("slope", max_slope)
    print("num_levels", num_levels)


    contours = density_to_loops(Sol, num_levels, x, y)

    fig, ax = plt.subplots(1,1)
    for c in contours:
        c.plot_contour(fig, ax)
    ax.set_aspect('equal')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    return contours

def CalcUniformityContours(contours, resolution, V = [-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
    x = np.linspace(V[0], V[1], resolution)  # Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    X = X.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
    Y = Y.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
    Z = Z.reshape((resolution ** 3), order="C")  # Order the volume lexicographically

    Bx = np.zeros_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)
    I = np.arange(len(X))
    for i, xv, yv, zv in tqdm(zip(I, X, Y, Z)):
        #print(i,xv, yv, zv)
        for c in contours:
            B = c.calc_mag_field(np.array([xv,yv,zv]))
            Bx[i] += B[0]
            By[i] += B[1]
            Bz[i] += B[2]
    
    Bx_avg = np.mean(Bx)
    By_avg = np.mean(By) 
    Bz_avg = np.mean(Bz)
    print(Bx_avg, By_avg, Bz_avg)  
    U = (np.var(Bx) + np.var(By) + np.var(Bz))/(Bx_avg**2+By_avg**2+Bz_avg**2) #not correct
    print(U)

    return Bx, By, Bz, U




if __name__ == "__main__":

    res = 10 #Must be even when using the symmetric matric constructors!
    N = 7
    M = N
    #u, V = TestGenAndOptPCB(N, M, res, tar_dir = "x", matrixGenerator = P_AssembleSSymRightOrder)
    #u, V = TestGenAndOptDoublePCB(N, M, res, matrixGenerator = P_AssembleSSymRightOrder)
    ulist, Vlist, Blist = TestReadAndOptPCB_Loop(N, M, res, tar_dir = "z", matrixGenerator = P_AssembleSSymRightOrder, solutionplots=True)

    plt.show()
    
