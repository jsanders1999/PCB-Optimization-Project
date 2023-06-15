import numpy as np
import scipy as sp
import scipy.integrate as si
import scipy.linalg as li
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from CubeClass import Cube
from DipoleFields import *
from LexgraphicTools import *
from PCBClass import PCB_u
from OptimizeClass import optimize_k
from Contour_new import *



def PlotSol(u, N, M, Res, V, B, tar_dir, fig = None, ax = None):
    """
    Plots the current potential and current loops for a PCB design.

    Args:
        u (numpy.ndarray): Solution vector.
        N (int): Grid Size in the x-direction.
        M (int): Grid Size in the y-direction.
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
        fig, ax = plt.subplots(1,1, figsize = (9,7))
    pc = ax.pcolormesh(X, Y, Sol, vmin = - max(np.max(Sol), np.min(Sol)), vmax = max(np.max(Sol), np.min(Sol)))
    ax.set_aspect('equal')
    fig.colorbar(pc)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Current Potential for N=M={}, Res={}, U={:.5f}, B_{}={:.4f}".format(2*int(N)+1+(tar_dir=="x"), Res, V, tar_dir, np.linalg.norm(B)))

    #fig2, ax2 = plt.subplots(1,1)
    #pc2 = ax2.contour(X, Y, Sol, levels = 20)
    #ax2.set_aspect('equal')
    #fig2.colorbar(pc2)
    #ax2.set_xlabel("x [m]")
    #ax2.set_ylabel("y [m]")
    #ax2.set_title("Current Loops for N={}, M={}, Res={}, U={:.5f}, B_z={:5f}".format(int(N), M, Res, V, np.linalg.norm(B)))
    #plt.show()
    return fig, ax


def PlotSolMinimal(u, tar_dir, fig = None, ax = None):
    """
    Plots the current potential and current loops for a PCB design.

    Args:
        u (numpy.ndarray): Solution vector.
        N (int): Grid Size in the x-direction.
        M (int): Grid Size in the y-direction.
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


def PlotSol3D_hatfunctions(u,pcb):
    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-1,1, 1000) 
    y = np.linspace(-1,1, 1000)
    X, Y = np.meshgrid(x, y)
    Sol = np.zeros(X.shape)

    X, Y = np.meshgrid(x, y)
    phi = np.vectorize(pcb.curl_potential)
    Phi = phi(X,Y)

    # Plot the flat plane on the XY plane
    ax.plot_surface(X, Y, Phi, facecolors=plt.cm.viridis((Phi -np.min(Phi))/(np.max(Phi)-np.min(Phi))), shade=False, antialiased=False, zorder=-1)

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

    x2 = np.linspace(V[0], V[1], pcb.cube.resolution)  # Discretize the target volume into a grid
    y2 = np.linspace(V[2], V[3], pcb.cube.resolution)
    z2 = np.linspace(V[4], V[5], pcb.cube.resolution)
    Z2, Y2, X2 = np.meshgrid(z2, y2, x2, indexing='ij')
    X2 = X2.reshape((pcb.cube.resolution ** 3), order="C")  # Order the volume lexicographically
    Y2 = Y2.reshape((pcb.cube.resolution ** 3), order="C")  # Order the volume lexicographically
    Z2 = Z2.reshape((pcb.cube.resolution ** 3), order="C")  # Order the volume lexicographically

    Bx = pcb.S_x@u
    By = pcb.S_y@u
    Bz = pcb.S_z@u
    B_max = max(np.linalg.norm(np.array([Bx, By, Bz]), axis = 1))
    length = 10/pcb.cube.resolution


    ax.quiver(X2, Y2, Z2, length*Bx/B_max, length*By/B_max, length*Bz/B_max, normalize=False, color = "k")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    #ax.view_init(elev=48, azim=-65)
    ax.view_init(elev=0, azim=-90)
    
    return fig, ax


def MultiPlot(startM, endM,f,tar_dir, res =25,save_path = 0):
    cube = Cube(res)
    nplots= endM-startM
    if nplots%4 != 0:
        print("nplots not divisible by 4, possibly not all shown")
    if nplots//4 ==1:
        print("Use more plots to work well")
    fig, axs = plt.subplots(nrows= max(2,nplots//4),ncols = 4, layout="constrained",figsize = (16,16))
    M= startM
    for row in range(nplots//4):
        for col in range(4):
            pcb = PCB_u(M, None, cube, 0)

            pcb.assemble_S()
            Q = pcb.SS_x + pcb.SS_y + pcb.SS_z
            J = np.ones((pcb.cube.resolution**3,pcb.cube.resolution**3))

            if tar_dir == "x":
                u = li.solve(Q,f*pcb.S_x.T@np.ones(res**3))

            if tar_dir == "z":
                u = li.solve(Q,f*pcb.S_z.T@np.ones(res**3))
           


            x = np.linspace(-1,1, 10000)
            y = np.linspace(-1,1, 10000)
            pcb.u_cart = np.reshape(u, (M,M), order = "C")

            pcb.coeff_to_current(20)

            X, Y = pcb.X_curr, pcb.Y_curr
            # phi = np.vectorize(pcb.potential)
            Phi = pcb.potential

            pcb.calc_mag_field()
            divnorm=colors.TwoSlopeNorm(vmin=min(np.min(Phi),-np.max(Phi)), vcenter=0., vmax=max(np.max(Phi),-np.min(Phi)))
            ax = axs[row,col]
            pc = ax.pcolormesh(X,Y, Phi,norm=divnorm)
            fig.colorbar(pc,ax = ax,fraction=0.046, pad=0.04)
            title_string = str(M) + " hat functions"
            ax.set_title(title_string)
            ax.set_aspect("equal")
            ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False,bottom=False)
            M +=1
    if save_path:
        plt.savefig(save_path+"Different_M_Grid_Plot_"+tar_dir,dpi = 600)

def MultiPlot_contour(startM, endM, tar_dir, seperation = 0.0003, f = 1e5, res =25,save_path = 0):
    cube = Cube(res)
    nplots= endM-startM
    if nplots%4 != 0:
        print("nplots not divisible by 4, possibly not all shown")
    if nplots//4 ==1:
        print("Use more plots to work well")
    fig, axs = plt.subplots(nrows= max(2,nplots//4),ncols = 4, layout="constrained",figsize = (16,16))
    M= startM
    for row in range(nplots//4):
        for col in range(4):
            pcb = PCB_u(M, None, cube, 0)

            pcb.assemble_S()
            Q = pcb.SS_x + pcb.SS_y + pcb.SS_z
            J = np.ones((pcb.cube.resolution**3,pcb.cube.resolution**3))
            if tar_dir == "x":
                u = li.solve(Q,f*pcb.S_x.T@np.ones(res))

            if tar_dir == "z":
                u = li.solve(Q,f*pcb.S_z.T@np.ones(res))

            pcb.u_cart = np.reshape(u, (M,M), order = "C")

            CalcContoursHat(pcb, tar_dir,)
            ##IF NEEDED THIS MUST BE CHANGED
            
            x = np.linspace(-1,1, 1000) 
            y = np.linspace(-1,1, 1000)

            X, Y = np.meshgrid(x, y)
            phi = np.vectorize(pcb.curl_potential)
            Phi = phi(X,Y)     
            PhiGrad = np.array(np.gradient(Phi))
            dx = x[1]-x[0]
            max_slope = np.max(np.linalg.norm(PhiGrad, axis = 0))
            num_levels = int((np.max(Phi)-np.min(Phi))/(max_slope/dx*seperation)) #(np.max(Sol)-np.min(Sol))/min_spacing*SolGrad
            # num_levels = 10
            print((np.max(Phi)-np.min(Phi)),(max_slope/dx*seperation) )
            print("slope", max_slope)
            print("num_levels", num_levels)
            print("I", max_slope/dx*seperation)
            ax = axs[row,col]
            contours = density_to_loops(Phi, num_levels, x, y)
            for c in contours:
                c.plot_contour(fig, ax)
            title_string = str(M) + " hat functions"
            ax.set_title(title_string)
            ax.set_aspect("equal")
            ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False,bottom=False)
            M +=1
    if save_path:
        plt.savefig(save_path+"Different_M_Countours_"+tar_dir,dpi = 600)



def UnifMagFieldPower_xz(startM, endM, f = 10**7,res = 25,save_path = 0):
    cube = Cube(res)
    Ms = range(startM,endM+1)
    Vs_x = []
    Vs_z = []
    Bs_x = []
    Bs_z = []
    Ps_x = []
    Ps_z = []
    for M in Ms:
        pcb = PCB_u(M, None, cube, 0)

        pcb.assemble_S()
        pcb.assemble_A()

        Q = pcb.SS_x + pcb.SS_y + pcb.SS_z

        u_x = li.solve(Q,f*pcb.S_x.T@np.ones(res**3))
        u_z = li.solve(Q,f*pcb.S_z.T@np.ones(res**3))

        Ju_x = np.concatenate((pcb.S_x@u_x- f*np.ones(res**3),pcb.S_y@u_x,pcb.S_z@u_x))
        Vs_x.append((1/f)**2*(1/res**3)*np.linalg.norm(Ju_x)**2)
        Bs_x.append(1/f*1/res**3*np.sum(pcb.S_x@u_x))
        Ps_x.append(1e-7*u_x@pcb.A@u_x)

        Ju_z = np.concatenate((pcb.S_x@u_z,pcb.S_y@u_z,pcb.S_z@u_z- f*np.ones(res**3)))
        Vs_z.append((1/f)**2*(1/res**3)*np.linalg.norm(Ju_z)**2)
        Bs_z.append(1/f*1/res**3*np.sum(pcb.S_z@u_z))
        Ps_z.append(1e-7*u_z@pcb.A@u_z)
    
    xticks = range(Ms[0],Ms[-1]+1,2)
    fig, ax = plt.subplots(nrows = 1,ncols = 3,figsize = (15,5),layout="constrained")
    ax[0].plot(Ms,Vs_x, linestyle = '--', marker = '.', label = "x-direction")
    ax[0].plot(Ms,Vs_z, linestyle = '--', marker = '.', label = "z-direction")
    ax[0].set_xticks(xticks)
    ax[0].set_ylabel("Non-uniformity [-]")
    ax[0].set_xlabel("Grid Size")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(Ms,Bs_x,linestyle = '--', marker = '.', label = "x-direction")
    ax[1].plot(Ms,Bs_z, linestyle = '--', marker = '.', label = "z-direction")
    ax[0].set_xticks(xticks)
    ax[1].set_xlabel("Grid Size")
    ax[1].set_ylabel("Average B field [mT]")
    ax[1].legend()
    ax[1].grid()
    ax[2].plot(Ms,Ps_x,linestyle = '--', marker = '.', label = "x-direction")
    ax[2].plot(Ms,Ps_z, linestyle = '--', marker = '.', label = "z-direction")
    ax[0].set_xticks(xticks)
    ax[2].set_xlabel("Grid Size")
    ax[2].set_ylabel("Power dissapated [W]") #check this!
    ax[2].set_yscale("log")
    ax[2].legend()
    ax[2].grid()

    if save_path:
        plt.savefig(save_path+"Non_uni_Mag_field_Power",dpi = 400)

def UnifMagFieldPower_OptivsWired(startM, endM, dir, seprs = [0.0025], res = 25, save_path = 0):
    cube = Cube(res)
    Ms = range(startM,endM+1)
    Vs_opti = []
    Bs_opti = []
    Ps_opti = []
    Vs_wire = np.zeros((len(seprs),len(Ms)))
    Bs_wire = np.zeros((len(seprs),len(Ms)))
    Ps_wire = np.zeros((len(seprs),len(Ms)))
    count = 0
    for M in Ms:
        pcb = PCB_u(M, None, cube, 0)

        pcb.assemble_S()
        Q = pcb.SS_x + pcb.SS_y + pcb.SS_z
        J = np.ones((pcb.cube.resolution**3,pcb.cube.resolution**3))
        if dir == "x":
            R = pcb.S_x.T@J@pcb.S_x
            u = li.solve(Q,pcb.S_x.T@np.ones(res**3))
        if dir == "z":
            R = pcb.S_z.T@J@pcb.S_z
            u = li.solve(Q,pcb.S_z.T@np.ones(res**3))

        Vs_opti.append((res**3*u.T@Q@u)/(u.T@R@u) - 1)
        Bs_opti.append(np.mean([1/res**3*np.sum(pcb.S_x@u),1/res**3*np.sum(pcb.S_y@u),1/res**3*np.sum(pcb.S_z@u)]))
        Ps_opti.append(u@u)
        
        for k in range(len(seprs)):
            contours, I = CalcContoursHat(pcb, seperation = seprs[k],savepath = save_path, plot = 1)

            Bx, By, Bz, U = CalcUniformityContours(contours, res,dir, I, V = [-0.5, 0.5, -0.5, 0.5, 0.1, 1.1])

            Vs_wire[k,count] = U
            Bs_wire[k,count] = np.mean([Bx,By,Bz])
            Ps_wire[k,count] = u@u
        count += 1

    
    xticks = range(Ms[0],Ms[-1],2)
    fig, ax = plt.subplots(nrows = 1,ncols = 2,figsize = (8,5),layout="constrained")
    # fig.suptitle("Non-Uniformity and Magnetic field of hat-functions")
    ax[0].plot(Ms,Vs_opti, linestyle = '--', marker = '.', label = "optimal")
    for k in range(len(seprs)):
        ax[0].plot(Ms,Vs_wire[k,:], linestyle = '--', marker = '.', label = "wires {} m".format(seprs[k]))
    ax[0].set_ylabel("Non-Uniformity")
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Number of hat-functions")
    ax[0].set_xticks(xticks)
    ax[0].legend()
    ax[1].plot(Ms,Bs_opti,linestyle = '--', marker = '.', label = "optimal")
    for k in range(len(seprs)):
        ax[1].plot(Ms,Bs_wire[k,:], linestyle = '--', marker = '.', label = "wires {} m".format(seprs[k]))
    ax[1].set_xlabel("Number of hat-functions")
    ax[1].set_xticks(xticks)
    ax[1].set_ylabel("Magnetic Field")
    ax[1].set_yscale('log')
    ax[1].legend()
    # ax[2].plot(Ms,Ps_opti,linestyle = '--', marker = '.', label = "optimal")
    # for k in range(len(seprs)):
    #     ax[2].plot(Ms,Ps_wire[k,:], linestyle = '--', marker = '.', label = "wires {} m".format(seprs[k]))
    # ax[2].set_xlabel("Number of hat-functions")
    # ax[2].set_xticks(xticks)
    # ax[2].set_ylabel("Power")
    # ax[2].set_yscale('log')
    # ax[2].legend()
    # if save_path:
    #     plt.savefig(save_path+"Non_uni_Mag_field_Power_OptivsWire",dpi = 400)


def plot_other(pcb,u,save_path,V,B, toplot = [True,True,True,True,True]):
    pcb.coeff_to_current(10)
    ### Figure 1 Imshow u
    if toplot[0]:
        fig1, ax1 = plt.subplots(1,1)

        pc2 = ax1.pcolormesh(pcb.X, pcb.Y, pcb.u_cart)
        plt.colorbar(pc2)
        ax1.set_title("Coefficients")
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax1.set_aspect("equal")
        if save_path:
            plt.savefig(save_path+"Imshow_u",dpi = 400)

    ##Figure 2 Potential
    if toplot[1]:
        fig2, ax2 = plt.subplots(1,1)
        pc3 = plt.pcolormesh(pcb.X_curr, pcb.Y_curr, pcb.potential)
        plt.colorbar(pc3)
        ax2.set_title("Potential")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        ax2.set_aspect("equal")
        if save_path:
            plt.savefig(save_path+"Potential",dpi = 400)

    ##Figure 3 Curl Potential colormesh (visible tents)
    if toplot[2]:

        # pcb.coeff_to_current(20)
        # pcb.plot_curl_potential(uniformity= V, B_tot = B, save_path = save_path)

        X, Y = pcb.X_curr, pcb.Y_curr
        # phi = np.vectorize(pcb.potential)
        Phi = pcb.potential
        fig, ax = plt.subplots(1,1)
        pc = ax.pcolormesh(X,Y, Phi)
        fig.colorbar(pc,ax = ax,fraction=0.046, pad=0.04)
        title_string = "Current Potential for N = {0}, res = {1},\n V = {2:.3e}, B = {3:.3e}".format(pcb.M, pcb.cube.resolution, V,B)
        ax.set_title(title_string)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.set_aspect("equal")
        if save_path:
            plt.savefig(save_path+"Curl_Potential",dpi = 400)


    ## Figure 4 Current Loops 
    if toplot[3]:    
        pcb.PlotSol3D_hatfunctions(u,save_path, 2)
        pcb.PlotSol3D_hatfunctions(u,save_path, 1)
    if toplot[4]:
        pcb.plot_field_arrow_extra([0.25,0.25,0.25,0.5,0.5,0.5,0.75,0.75,0.75],[0,1,2,0,1,2,0,1,2],save_path = save_path)
        pcb.plot_field_arrow_all_orien([0.25,0.25,0.25],save_path = save_path)
