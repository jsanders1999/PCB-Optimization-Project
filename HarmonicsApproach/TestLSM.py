from time import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from Streamfunctions import *
from MatrixConstructors import *
from ParallelMatrixConstructors import *
from Optimize import *
from ReadWriteArray import *
from PlottingMethods import *
from Contour import *

plt.style.use("seaborn-v0_8-dark")

trace_dist_arr = [ 0.01, 0.003, 0.001, 0.0003]
FACTOR = 1e7*1e-3 #1 millitestla target
rho = 1.72e-8
delh = 70e-6


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
    if tar_dir == "x":
        u_ls = sp.linalg.solve(Q, Sx.T@np.ones((res**3)), assume_a = "pos" )
        r_ls = np.linalg.norm(np.ones((res**3)) - Sx@u_ls)
    elif tar_dir == "z":
        u_ls = sp.linalg.solve(Q, Sz.T@np.ones((res**3)), assume_a = "pos" )
        r_ls = np.linalg.norm(np.ones((res**3)) - Sz@u_ls)

    #u_ls = u_ls/np.linalg.norm(u_ls)
    B_ls = [1/res**3*np.sum(Sx@u_ls), 1/res**3*np.sum(Sy@u_ls), 1/res**3*np.sum(Sz@u_ls)]

    #Plotting
    fig3, ax3 = PlotSol(u_ls, N, M, res, r_ls, B_ls, tar_dir)
    fig4, ax4 = plt.subplots(1,1)
    ax4.imshow(u_ls.reshape((N,M)))
    
    return u_ls, r_ls

def TestReadAndOptPCB_Loop(N2, M2, res, tar_dir, matrixGenerator = P_AssembleS, solutionplots = False, calcContours = True):
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
    Vlist = [] #Variance relative to target for surface current
    #Ulist = [] #Variance relatice to target for current loops
    Blist = [] #Average magnetic field
    Plist = [] #Power use, TODO: Put in correct units

    N_max = min(13, N2+1)
    
    U_arr =         np.zeros( (len(trace_dist_arr), N_max-1)  )
    I_arr =         np.zeros( (len(trace_dist_arr), N_max-1)  )
    trace_num_arr = np.zeros( (len(trace_dist_arr), N_max-1)  )

    

    for N1 in range(1, N_max):
        print()
        print("ORDER: ", N1)
        M1 = N1
        Sx = extract_subarray(Sx_large, N1, M1, N2, M2)
        Sy = extract_subarray(Sy_large, N1, M1, N2, M2)
        Sz = extract_subarray(Sz_large, N1, M1, N2, M2)
        

        Q = Sx.T@Sx + Sy.T@Sy + Sz.T@Sz
        if tar_dir == "z":
            ind = 2
        elif tar_dir == "x":
            ind = 0


        #optimization
        if tar_dir == "x":
            u_ls = FACTOR*sp.linalg.solve(Q, Sx.T@np.ones((res**3)))#, assume_a = "pos")

            #L = sp.linalg.cholesky(Q)
            #y = sp.linalg.solve(L, Sx.T@np.ones((res**3)) )
            #u_ls = sp.linalg.solve(L.T, y)#, assume_a = "pos")
            r_ls = (np.linalg.norm(Sz@u_ls)**2 + np.linalg.norm(Sy@u_ls)**2 + np.linalg.norm(FACTOR*np.ones((res**3)) - Sx@u_ls)**2)/(res**3*FACTOR**2)

        elif tar_dir == "z":
            u_ls = FACTOR*sp.linalg.solve(Q, Sz.T@np.ones((res**3)))#, assume_a = "pos")
            #L = sp.linalg.cholesky(Q)
            #y = sp.linalg.solve(L, Sz.T@np.ones((res**3)) )
            #u_ls = sp.linalg.solve(L.T, y)#, assume_a = "pos")
            r_ls = (np.linalg.norm(Sx@u_ls)**2 + np.linalg.norm(Sy@u_ls)**2 + np.linalg.norm(FACTOR*np.ones((res**3)) - Sz@u_ls)**2)/(res**3*FACTOR**2)

        solutionlist +=[u_ls]
        Vlist += [r_ls]
        B_ls = [1/res**3*np.sum(Sx@u_ls), 1/res**3*np.sum(Sy@u_ls), 1/res**3*np.sum(Sz@u_ls)]
        Blist += [B_ls]
        Plist += [rho/delh*u_ls.T@u_ls]
        print("Order:" , N1, r_ls, B_ls[ind], u_ls.T@u_ls)

        #Plotting
        if solutionplots or N1 == 7: #and N1%2 == 0:
            fig3, ax3 = PlotSol(u_ls, N1, M1, res, r_ls, B_ls, tar_dir)
            #fig3, ax3 = PlotSol3D(u_ls, Sx, Sy, Sz, tar_dir, N, M, res)

        if calcContours:
            for t, trace_dist in enumerate(trace_dist_arr):
                print( "trace dist: ", trace_dist)
                contours, I, trace_num = CalcContoursHarmonics(u_ls, tar_dir, N1, M1, seperation = trace_dist, showplots=(N1==7))
                Bx, By, Bz, U = CalcUniformityContours(contours, res, I, tar_dir)
                U_arr[t, N1-1] = U
                I_arr[t, N1-1] = I
                trace_num_arr[t, N1-1] = trace_num

    return solutionlist, Vlist, Blist, Plist, U_arr, I_arr, trace_num_arr

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
        Sx_large_z = read_array_from_file( "HarmonicsApproach/Matrices/Sx_z_N{}_M{}_res{}_Gen_{}.txt".format( N, M, res, matrixGenerator.__name__) )
        Sy_large_z = read_array_from_file( "HarmonicsApproach/Matrices/Sy_z_N{}_M{}_res{}_Gen_{}.txt".format( N, M, res, matrixGenerator.__name__) )
        Sz_large_z = read_array_from_file( "HarmonicsApproach/Matrices/Sz_z_N{}_M{}_res{}_Gen_{}.txt".format( N, M, res, matrixGenerator.__name__) )
        Sx_large_x = read_array_from_file( "HarmonicsApproach/Matrices/Sx_x_N{}_M{}_res{}_Gen_{}.txt".format( N, M, res, matrixGenerator.__name__) )
        Sy_large_x = read_array_from_file( "HarmonicsApproach/Matrices/Sy_x_N{}_M{}_res{}_Gen_{}.txt".format( N, M, res, matrixGenerator.__name__) )
        Sz_large_x = read_array_from_file( "HarmonicsApproach/Matrices/Sz_x_N{}_M{}_res{}_Gen_{}.txt".format( N, M, res, matrixGenerator.__name__) )
    except:
        print("File does not exist, generate the matrix first")

    solutionlist_x = []
    solutionlist_z = []
    Vlist_x = []
    Vlist_z = []
    Blist_x = []
    Blist_z = []

    for N1 in range(1,N2+1):
        M1 = N1
        Sx_x = extract_subarray(Sx_large_x, N1, M1, N2, M2)
        Sy_x = extract_subarray(Sy_large_x, N1, M1, N2, M2)
        Sz_x = extract_subarray(Sz_large_x, N1, M1, N2, M2)
        Sx_z = extract_subarray(Sx_large_z, N1, M1, N2, M2)
        Sy_z = extract_subarray(Sy_large_z, N1, M1, N2, M2)
        Sz_z = extract_subarray(Sz_large_z, N1, M1, N2, M2)

        Q_x = Sx_x.T@Sx_x + Sy_x.T@Sy_x + Sz_x.T@Sz_x
        Q_x = Sx_z.T@Sx_z + Sy_z.T@Sy_z + Sz_z.T@Sz_z


        #optimization
        
        u_ls_x = sp.linalg.solve(Q, Sx_x.T@np.ones((res**3)))#, assume_a = "pos")
        r_ls_x = np.linalg.norm(np.ones((res**3)) - Sx_x@u_ls_x)/res**3
       
        u_ls_z = sp.linalg.solve(Q, Sz_z.T@np.ones((res**3)))#, assume_a = "pos")
        r_ls_z = np.linalg.norm(np.ones((res**3)) - Sz_z@u_ls)/res**3

        solutionlist_x += [u_ls_x]
        solutionlist_z += [u_ls_z]
        Vlist_x += [r_ls_x]
        Vlist_z += [r_ls_z]
        B_ls_x = [1/res**3*np.sum(Sx_x@u_ls_x), 1/res**3*np.sum(Sy_x@u_ls_x), 1/res**3*np.sum(Sz_x@u_ls_x)]
        B_ls_z = [1/res**3*np.sum(Sx_z@u_ls_z), 1/res**3*np.sum(Sy_z@u_ls_z), 1/res**3*np.sum(Sz_z@u_ls_z)]
        Blist_x += [B_ls_x]
        Blist_z += [B_ls_z]
        print(N1, u_ls_x.T@u_ls_x, u_ls_z.T@u_ls_z )

        #Plotting
        if solutionplots:
            
            fig3, ax3 = PlotSol(u_ls, N1, M1, res, r_ls, B_ls, tar_dir)
            #fig3, ax3 = PlotSol3D(u_ls, Sx, Sy, Sz, tar_dir, N, M, res)
            #fig4, ax4 = plt.subplots(1,1)
            #ax4.imshow(u_opt.reshape((N1,M1)))
            #PlotSol3D(u_opt, Sx, Sy, Sz)
            #contours = CalcContoursHarmonics(u_opt, tar_dir, N, M)
            #CalcUniformityContours(contours, res)
            


    fig5, ax5= plt.subplots(2,1)
    ax5[0].plot(range(1,N2+1),Vlist, marker = ".", linestyle = "None")
    ax5[1].plot(range(1,N2+1),np.array(Blist)[:,ind],  marker = ".", linestyle = "None")
    #ax5[0].set_xlabel("Number of harmonics")
    ax5[0].set_ylabel("Non-uniformity, r")
    ax5[0].set_yscale("log")
    #ax5.set_ylim((0, 1.2*max(Vlist)))
    ax5[1].set_xlabel("Number of harmonics")
    ax5[1].set_ylabel("Magnetic field in {} direction".format(tar_dir))
    ax5[1].set_yscale("log")

    
    return solutionlist_x, solutionlist_z, Vlist, Blist

def PlotMetrics(ulists, Vlists, Blists, Plists, dirs = ["x", "z"], inds = [0, 2]):
    offset = {"x":1, "z":0}
    fig5, ax5= plt.subplots(1,3, figsize = (15,5))
    for i, Vlist in enumerate(Vlists):
        ax5[0].plot(range(1+offset[dirs[i]], 2*len(Vlist)+1+offset[dirs[i]], 2), Vlist                      ,   marker = ".", linestyle = "--", label = "{} target".format(dirs[i]))
    for i, Blist in enumerate(Blists):
        ax5[1].plot(range(1+offset[dirs[i]], 2*len(Blist)+1+offset[dirs[i]], 2), np.array(Blist)[:,inds[i]]/FACTOR,  marker = ".", linestyle = "--", label = "{} target".format(dirs[i]))
    for i, Plist in enumerate(Plists):
        ax5[2].plot(range(1+offset[dirs[i]], 2*len(Plist)+1+offset[dirs[i]], 2), Plist                      ,  marker = ".", linestyle = "--", label = "{} target".format(dirs[i]))
    ax5[0].set_ylabel("Non-uniformity [-]")
    ax5[0].set_xlabel("Number of harmonics")
    ax5[0].set_yscale("log")
    ax5[0].legend()
    ax5[0].grid()
    #ax5.set_ylim((0, 1.2*max(Vlist)))
    ax5[1].set_xlabel("Number of harmonics")
    ax5[1].set_ylabel("Average B field [mT]")
    ax5[1].legend()
    ax5[1].grid()


    ax5[2].set_xlabel("Number of harmonics")
    ax5[2].set_ylabel("Power dissapated [W]") #check this!
    ax5[2].set_yscale("log")
    ax5[2].legend()
    ax5[2].grid()

    return

def PlotSolutionsGrid(ulist, tar_dir ):
    plt.style.use("default")
    fig, ax = plt.subplots(3,4, figsize = (13,7.5))

    if tar_dir=="z":
        HarmStreamFunc = HarmStreamFunc_Ztar
    elif tar_dir == "x":
        HarmStreamFunc = HarmStreamFunc_Xtar
    x = np.linspace(-1,1, 200) 
    y = np.linspace(-1,1, 200)
    X, Y = np.meshgrid(x, y)
    
    for j, u in enumerate(ulist):
        Sol = np.zeros(X.shape)
        N = np.sqrt(u.size)
        for i, c in enumerate(u):
            m = i//N
            n = i%N
            Sol += c*HarmStreamFunc(X, Y, n, m)
        print(j, j//4, j%4)

        pc = ax[j//4, j%4].pcolormesh(X, Y, Sol, vmin = - max(np.max(Sol), np.min(Sol)), vmax = max(np.max(Sol), np.min(Sol)), )
        ax[j//4, j%4].set_aspect('equal')
        if 2*j+1+(tar_dir=="x") == 1:
            ax[j//4, j%4].set_title('{} harmonic'.format(2*j+1+(tar_dir=="x")))
        else:
            ax[j//4, j%4].set_title('{} harmonics'.format(2*j+1+(tar_dir=="x")))
        ax[j//4, j%4].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
        fig.colorbar(pc)
    return

def PlotLineSolutionsGrid(ulist, tar_dir, seperation ):
    fig, ax = plt.subplots(3,4, figsize = (13,7.5))

    if tar_dir=="z":
        HarmStreamFunc = HarmStreamFunc_Ztar
    elif tar_dir == "x":
        HarmStreamFunc = HarmStreamFunc_Xtar
    #x = np.linspace(-1,1, 200) 
    #y = np.linspace(-1,1, 200)
    #X, Y = np.meshgrid(x, y)
    
    for j, u in enumerate(ulist):
        x = np.linspace(-1,1, 1000) 
        y = np.linspace(-1,1, 1000)
        X, Y = np.meshgrid(x, y)
        Sol = np.zeros(X.shape)
        N = np.sqrt(len(u))

        for i, a in enumerate(u):
            m = i//N
            n = i%N
            Sol += a*HarmStreamFunc(X, Y, n, m)
     
        SolGrad = np.array(np.gradient(Sol))
        dx = x[1]-x[0]
        max_slope = np.max(np.linalg.norm(SolGrad, axis = 0))
        num_levels = int((np.max(Sol)-np.min(Sol))/(max_slope/dx*seperation)) #(np.max(Sol)-np.min(Sol))/min_spacing*SolGrad
        print("num_levels :", num_levels, " I :", max_slope/dx*seperation)

        contours = density_to_loops(Sol, num_levels, x, y)

        for c in contours:
            c.plot_contour(fig, ax[j//4, j%4])
        
        
        ax[j//4, j%4].set_aspect('equal')
        if 2*j+1+(tar_dir=="x") == 1:
            ax[j//4, j%4].set_title('{} harmonic'.format(2*j+1+(tar_dir=="x")))
        else:
            ax[j//4, j%4].set_title('{} harmonics'.format(2*j+1+(tar_dir=="x")))
        ax[j//4, j%4].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    return

def CalcContoursHarmonics(u, tar_dir, N, M, seperation = 0.0005,  showplots = False):
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
    SolGrad = np.array(np.gradient(Sol))
    dx = x[1]-x[0]
    max_slope = np.max(np.linalg.norm(SolGrad, axis = 0))
    num_levels = int((np.max(Sol)-np.min(Sol))/(max_slope/dx*seperation)) #(np.max(Sol)-np.min(Sol))/min_spacing*SolGrad
    print("num_levels :", num_levels, " I :", max_slope/dx*seperation)
    I = max_slope/dx*seperation

    contours = density_to_loops(Sol, num_levels, x, y)

    if showplots:
        fig, ax = plt.subplots(1,1, figsize = (8,6))
        for c in contours:
            c.plot_contour(fig, ax)
        ax.set_aspect('equal')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title('{} harmonics, d = {} mm'.format(2*N+1+(tar_dir=="x"), 1000*seperation))

    return contours, I, num_levels

def CalcUniformityContours(contours, resolution, I, tar_dir,  V = [-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
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
    Ind = np.arange(len(X))
    for i in tqdm(Ind):
        for c in contours:
            B = I*c.calc_mag_field(np.array([X[i], Y[i], Z[i]]))
            Bx[i] += B[0]
            By[i] += B[1]
            Bz[i] += B[2]
    
    Bx_avg = np.mean(Bx)
    By_avg = np.mean(By) 
    Bz_avg = np.mean(Bz)
    print("Magnetic fields for lines: ", Bx_avg, By_avg, Bz_avg) 
    if tar_dir == "z":
        U = (np.sum((Bx)**2) + np.sum((By)**2) + np.sum((Bz-FACTOR )**2))/(res**3*FACTOR**2)
    if tar_dir == "x":
        U = (np.sum((Bx-FACTOR )**2) + np.sum((By)**2) + np.sum((Bz)**2))/(res**3*FACTOR**2)
    print("U lines: ", U)
    
    if False: 
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        x2 = np.linspace(-0.5, 0.5, resolution)  # Discretize the target volume into a grid
        y2 = np.linspace(-0.5, 0.5, resolution)
        z2 = np.linspace(0.1, 1.1, resolution)
        Z2, Y2, X2 = np.meshgrid(z2, y2, x2, indexing='ij')
        X2 = X2.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
        Y2 = Y2.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
        Z2 = Z2.reshape((resolution ** 3), order="C")  # Order the volume lexicographically

        B_max = max(np.linalg.norm(np.array([Bx, By, Bz]), axis = 1))
        length = 10/resolution


        ax.quiver(X2, Y2, Z2, length*Bx/B_max, length*By/B_max, length*Bz/B_max, normalize=False, color = "k")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        #ax.view_init(elev=48, azim=-65)
        ax.view_init(elev=0, azim=-90)

        plt.show()

    return Bx, By, Bz, U

def PlotMetricsLines(Vlist, U_arr, I_arr, trace_num_arr, dir, spacing = trace_dist_arr):
    offset = {"x":1, "z":0}
    fig5, ax5= plt.subplots(1,3, figsize = (15,5))
    
    ax5[0].plot(range(1+offset[dir], 2*len(Vlist)+1+offset[dir], 2), Vlist          ,   marker = ".", linestyle = "-", color = "k", label = "Plane current")
    for i, Ulist in enumerate(U_arr):
        print(Ulist)
        ax5[0].plot(range(1+offset[dir], 2*len(Ulist)+1+offset[dir], 2), Ulist     ,   marker = ".", linestyle = "--", label = "Traces for d = {} mm".format(1000*spacing[i]))

    ax5[0].set_ylabel("Non-uniformity [-]")
    ax5[0].set_xlabel("Number of harmonics")
    ax5[0].set_yscale("log")
    ax5[0].legend()
    ax5[0].grid()
    ax5[0].set_ylim(top=10)

    for i, trace_num_list in enumerate(trace_num_arr):
        ax5[1].plot(range(1+offset[dir], 2*len(trace_num_list)+1+offset[dir], 2), trace_num_list,   marker = ".", linestyle = "--", label = "Traces for d = {} mm".format(1000*spacing[i]))
    
    ax5[1].set_xlabel("Number of harmonics")
    ax5[1].set_ylabel("Number of contour lines")
    ax5[1].legend()
    ax5[1].grid()

    for i, Ilist in enumerate(I_arr):
        ax5[2].plot(range(1+offset[dir], 2*len(Ilist)+1+offset[dir], 2), Ilist     ,   marker = ".", linestyle = "--", label = "Traces for d = {} mm".format(1000*spacing[i]))

    ax5[2].set_xlabel("Number of harmonics")
    ax5[2].set_ylabel("Current per trace [A]")
    ax5[2].set_yscale("log")
    ax5[2].legend()
    ax5[2].grid()


    return



if __name__ == "__main__":

    res = 22 #Must be even when using the symmetric matric constructors!
    N = 18
    M = N
    calcContours = False
    
    #u, V = TestGenAndOptPCB(N, M, res, tar_dir = "x", matrixGenerator = P_AssembleSSymRightOrder)
    #u, V = TestGenAndOptPCB(N, M, res, tar_dir = "z", matrixGenerator = P_AssembleSSymRightOrder)

    ulist_x, Vlist_x, Blist_x, Plist_x, U_arr_x, I_arr_x, trace_num_arr_x = TestReadAndOptPCB_Loop(N, M, res, tar_dir = "x", matrixGenerator = P_AssembleSSymRightOrder, solutionplots=False, calcContours = calcContours)
    ulist_z, Vlist_z, Blist_z, Plist_z, U_arr_z, I_arr_z, trace_num_arr_z = TestReadAndOptPCB_Loop(N, M, res, tar_dir = "z", matrixGenerator = P_AssembleSSymRightOrder, solutionplots=False, calcContours = calcContours)
    PlotLineSolutionsGrid(ulist_z, "z", 0.003)
    PlotLineSolutionsGrid(ulist_x, "x", 0.003)

    PlotMetrics((ulist_x, ulist_z), (Vlist_x, Vlist_z), (Blist_x, Blist_z), (Plist_x, Plist_z))
    
    #if calcContours:

    PlotMetricsLines(Vlist_z, U_arr_z, I_arr_z, trace_num_arr_z, dir = "z")
    PlotMetricsLines(Vlist_x, U_arr_x, I_arr_x, trace_num_arr_x, dir = "x")

    
    PlotSolutionsGrid(ulist_z, "z")
    PlotSolutionsGrid(ulist_x, "x")

    plt.show()
    