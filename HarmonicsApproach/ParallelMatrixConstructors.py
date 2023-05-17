import numpy as np
import scipy.integrate as si
from numba import njit, jit
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from MatrixConstructors import AssembleSx, AssembleSySurf

from Streamfunctions import *

##### Integrands to find B in x, y, z #####

@njit
def psiz_conv_delx_harm_integrand(s, t, x, y, z, n, m):
    return -psi_z(x-s, y-t, z)*delx_HarmStreamFunc(s,t,n,m)

@njit
def psiz_conv_dely_harm_integrand(s, t, x, y, z, n, m):
    return -psi_z(x-s, y-t, z)*dely_HarmStreamFunc(s,t,n,m)

@njit
def psixy_conv_delxy_harm_integrand(s, t, x, y, z, n, m):
    return psi_x(x-s, y-t, z)*delx_HarmStreamFunc(s,t,n,m) + psi_y(x-s, y-t, z)*dely_HarmStreamFunc(s,t,n,m)

#### B in x, y, z ####
def psiz_conv_dely_harm(x, y, z, n, m): #Bx
    return si.dblquad(psiz_conv_dely_harm_integrand, -1, 1, -1, 1, args = (x, y, z, n, m))[0]
#psiz_conv_dely_harm = np.vectorize(psiz_conv_dely_harm)

def psiz_conv_delx_harm(x, y, z, n, m): #By
    return si.dblquad(psiz_conv_delx_harm_integrand, -1, 1, -1, 1, args = (x, y, z, n, m))[0]
#psiz_conv_delx_harm = np.vectorize(psiz_conv_delx_harm)

def psixy_conv_delxy_harm(x, y, z, n, m): #Bz
    return si.dblquad(psixy_conv_delxy_harm_integrand, -1, 1, -1, 1, args = (x, y, z, n, m))[0]
#psixy_conv_delxy_harm = np.vectorize(psixy_conv_delxy_harm)


#### The system matrices (steam functions coeffficients |--> Magnetic field at grid points in target volume ) ####

def P_AssembleS(N, M, resolution, direction, Omega=[-1.0, 1.0, -1.0, 1.0], V=[-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
    """
    Assembles the S matrix for the magnetic field in a given direction in a three-dimensional volume.
    Uses a parallelization to speed up the computations of the for loop of the integrals.

    Args:
        N (int): Number of harmonics in the x-direction.
        M (int): Number of harmonics in the y-direction.
        resolution (int): Number of grid points in the target volume for each dimension (x, y, z).
        direction (str): Direction of the magnetic field corresponding to the S matrix ('x', 'y', or 'z').
        Omega (list, optional): Boundaries of the volume in the form [x_min, x_max, y_min, y_max, z_min, z_max].
                                Defaults to [-1.0, 1.0, -1.0, 1.0].
        V (list, optional): Boundaries of the target volume in the form [x_min, x_max, y_min, y_max, z_min, z_max].
                            Defaults to [-0.5, 0.5, -0.5, 0.5, 0.1, 1.1].

    Returns:
        numpy.ndarray: S matrix of shape (resolution ** 3, N * M) containing the computed values.

    Raises:
        ValueError: If the direction argument is not 'x', 'y', or 'z'.

    """
    NM = N * M
    x = np.linspace(V[0], V[1], resolution)  # Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
    Y = Y.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
    Z = Z.reshape((resolution ** 3), order="C")  # Order the volume lexicographically

    S = np.zeros((resolution ** 3, NM))

    if direction == "x":
        B = psiz_conv_delx_harm
    elif direction == "y":
        B = psiz_conv_dely_harm
    elif direction == "z":
        B = psixy_conv_delxy_harm
    else:
        raise ValueError("Invalid direction. Must be 'x', 'y', or 'z'.")

    def compute_s(i, NM, B):
        temp_s = np.zeros(NM)
        for j in range(NM):
            m = j // N
            n = j % N
            temp_s[j] = B(X[i], Y[i], Z[i], n, m)
        return temp_s

    S = np.array(
        Parallel(n_jobs=-1)(
            delayed(compute_s)(i, NM, B) for i in tqdm(range(resolution ** 3), f"Assembling S{direction} for the Full Volume")
        )
    )

    return S


#### The system matrices (steam functions coeffficients |--> Magnetic field at grid points in target volume ) ####

def P_AssembleSSymm(N, M, resolution, direction, Omega=[-1.0, 1.0, -1.0, 1.0], V=[-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
    """
    Assembles the S matrix for the magnetic field in a given direction in a three-dimensional volume.
    Uses a parallelization to speed up the computations of the for loop of the integrals.
    Uses symmetry of the system in the x-z and y-z panes to avoid repeating the same integral. 

    Note: Uses a different ordering for the volume grid points than the full volume method

    Args:
        N (int): Number of harmonics in the x-direction.
        M (int): Number of harmonics in the y-direction.
        resolution (int): Number of grid points in the target volume for each dimension (x, y, z).
        direction (str): Direction of the magnetic field corresponding to the S matrix ('x', 'y', or 'z').
        Omega (list, optional): Boundaries of the volume in the form [x_min, x_max, y_min, y_max, z_min, z_max].
                                Defaults to [-1.0, 1.0, -1.0, 1.0].
        V (list, optional): Boundaries of the target volume in the form [x_min, x_max, y_min, y_max, z_min, z_max].
                            Defaults to [-0.5, 0.5, -0.5, 0.5, 0.1, 1.1].

    Returns:
        numpy.ndarray: S matrix of shape (resolution ** 3, N * M) containing the computed values.

    Raises:
        ValueError: If the target volume is not above the center of the surface
        ValueError: If the direction argument is not 'x', 'y', or 'z'.
        ValueError: If the resolution is not even

    """
    if (V[0] + V[1]) / 2 != (Omega[0] + Omega[1]) / 2 or (V[2] + V[3]) / 2 != (Omega[2] + Omega[3]) / 2:
        raise ValueError("The box is not in the middle of the surface.")
    if resolution%2 !=0:
        raise ValueError("The resultion of {} is not even".format(resolution))
    NM = N * M
    x = np.linspace(V[0], V[1], resolution)[:int(resolution / 2)]  # Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)[:int(resolution / 2)]
    z = np.linspace(V[4], V[5], resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape((int(resolution ** 3 / 4)), order="C")  # Order the volume lexicographically
    Y = Y.reshape((int(resolution ** 3 / 4)), order="C")  # Order the volume lexicographically
    Z = Z.reshape((int(resolution ** 3 / 4)), order="C")  # Order the volume lexicographically

    S = np.zeros((int(resolution ** 3 / 4), NM))

    if direction == "x":
        B = psiz_conv_delx_harm
        concat_order = [1, 1, -1, -1]
    elif direction == "y":
        B = psiz_conv_dely_harm
        concat_order = [1, -1, 1, -1]
    elif direction == "z":
        B = psixy_conv_delxy_harm
        concat_order = [1, 1, 1, 1]
    else:
        raise ValueError("Invalid direction. Must be 'x', 'y', or 'z'.")

    def compute_s(i, NM, B):
        temp_s = np.zeros(NM)
        for j in range(NM):
            m = j // N
            n = j % N
            temp_s[j] = B(X[i], Y[i], Z[i], n, m)
        return temp_s

    S = np.array(
        Parallel(n_jobs=-1)(
            delayed(compute_s)(i, NM, B) for i in tqdm(range(int(resolution ** 3 / 4)), f"Assembling S{direction} for the Full Volume using symmetry")
        )
    )

    S_full = np.concatenate([concat_order[0] * S, concat_order[1] * S, concat_order[2] * S, concat_order[3] * S], axis=0)
    return S_full

def P_AssembleSSurf(N, M, resolution, direction, Omega=[-1.0, 1.0, -1.0, 1.0], V=[-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
    """
    Assembles the S matrix for the magnetic field in a given direction on the edge of the three-dimensional volume.
    Uses a parallelization to speed up the computations of the for loop of the integrals.

    Args:
        N (int): Number of harmonics in the x-direction.
        M (int): Number of harmonics in the y-direction.
        resolution (int): Number of grid points in the target volume for each dimension (x, y, z).
        direction (str): Direction of the magnetic field corresponding to the S matrix ('x', 'y', or 'z').
        Omega (list, optional): Boundaries of the volume in the form [x_min, x_max, y_min, y_max, z_min, z_max].
                                Defaults to [-1.0, 1.0, -1.0, 1.0].
        V (list, optional): Boundaries of the target volume in the form [x_min, x_max, y_min, y_max, z_min, z_max].
                            Defaults to [-0.5, 0.5, -0.5, 0.5, 0.1, 1.1].

    Returns:
        numpy.ndarray: S matrix of shape (resolution ** 3, N * M) containing the computed values.

    Raises:
        ValueError: If the direction argument is not 'x', 'y', or 'z'.

    """
    NM = N * M
    x = np.linspace(V[0], V[1], resolution)  # Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
    Y = Y.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
    Z = Z.reshape((resolution ** 3), order="C")  # Order the volume lexicographically

    S = np.zeros((resolution ** 3, NM))

    if direction == "x":
        B = psiz_conv_delx_harm
    elif direction == "y":
        B = psiz_conv_dely_harm
    elif direction == "z":
        B = psixy_conv_delxy_harm
    else:
        raise ValueError("Invalid direction. Must be 'x', 'y', or 'z'.")
    
    deletelist = []
    for i in range(1,resolution-1):
        for j in range(1,resolution-1):
            for k in range(1,resolution-1):
                deletelist += [resolution**2*i+resolution*j+k]
    X_new = np.delete(X, deletelist)
    Y_new = np.delete(Y, deletelist)
    Z_new = np.delete(Z, deletelist)

    def compute_s(i, NM, B):
        temp_s = np.zeros(NM)
        for j in range(NM):
            m = j // N
            n = j % N
            temp_s[j] = B(X_new[i], Y_new[i], Z_new[i], n, m)
        return temp_s

    S = np.array(
        Parallel(n_jobs=-1)(
            delayed(compute_s)(i, NM, B) for i in tqdm(range((resolution**3 -(resolution-2)**3)), f"Assembling S{direction} for the edge of the Volume")
        )
    )

    return S

def P_AssembleSSurfSymm(N, M, resolution, direction, Omega=[-1.0, 1.0, -1.0, 1.0], V=[-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
    """
    Assembles the S matrix for the magnetic field in a given direction on the edge of the three-dimensional volume.
    Uses a parallelization to speed up the computations of the for loop of the integrals.
    Uses symmetry of the system in the x-z and y-z panes to avoid repeating the same integral. 

    Note: Uses a different ordering for the volume grid points than the method that does not use symmetry

    Args:
        N (int): Number of harmonics in the x-direction.
        M (int): Number of harmonics in the y-direction.
        resolution (int): Number of grid points in the target volume for each dimension (x, y, z).
        direction (str): Direction of the magnetic field corresponding to the S matrix ('x', 'y', or 'z').
        Omega (list, optional): Boundaries of the volume in the form [x_min, x_max, y_min, y_max, z_min, z_max].
                                Defaults to [-1.0, 1.0, -1.0, 1.0].
        V (list, optional): Boundaries of the target volume in the form [x_min, x_max, y_min, y_max, z_min, z_max].
                            Defaults to [-0.5, 0.5, -0.5, 0.5, 0.1, 1.1].

    Returns:
        numpy.ndarray: S matrix of shape (resolution ** 3, N * M) containing the computed values.

    Raises:
        ValueError: If the target volume is not above the center of the surface
        ValueError: If the direction argument is not 'x', 'y', or 'z'.
        ValueError: If the resolution is not even
        
    """
    if (V[0] + V[1]) / 2 != (Omega[0] + Omega[1]) / 2 or (V[2] + V[3]) / 2 != (Omega[2] + Omega[3]) / 2:
        raise ValueError("The box is not in the middle of the surface.")
    if resolution%2 != 0:
        raise ValueError("The resultion of {} is not even".format(resolution))
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x)
    X = X.reshape((resolution**3), order = "C") #order the volume lexographically
    Y = Y.reshape((resolution**3), order = "C") #order the volume lexographically
    Z = Z.reshape((resolution**3), order = "C") #order the volume lexographically

    S = np.zeros((resolution ** 3, NM))

    if direction == "x":
        B = psiz_conv_delx_harm
        concat_order = [1, 1, -1, -1]
    elif direction == "y":
        B = psiz_conv_dely_harm
        concat_order = [1, -1, 1, -1]
    elif direction == "z":
        B = psixy_conv_delxy_harm
        concat_order = [1, 1, 1, 1]
    else:
        raise ValueError("Invalid direction. Must be 'x', 'y', or 'z'.")
    
    #remove the interior elements and the edge elements of the target volume
    deletelist = []
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                if (k ==0 or k==resolution-1) and j<resolution/2 and i< resolution/2:
                    deletelist += []
                elif j==0 and i<resolution/2:
                    deletelist += []
                elif i==0 and j<resolution/2:
                    deletelist += []
                else:
                    deletelist += [resolution**2*j+resolution*k+i]
    X_new = np.delete(X, deletelist)
    Y_new = np.delete(Y, deletelist)
    Z_new = np.delete(Z, deletelist)

    def compute_s(i, NM, B):
        temp_s = np.zeros(NM)
        for j in range(NM):
            m = j // N
            n = j % N
            temp_s[j] = B(X_new[i], Y_new[i], Z_new[i], n, m)
        return temp_s

    S = np.array(
        Parallel(n_jobs=-1)(
            delayed(compute_s)(i, NM, B) for i in tqdm(range(int((resolution**3 -(resolution-2)**3)/4)), f"Assembling S{direction} for the edge of the Volume using Symmetry")
        )
    )

    S_full = np.concatenate([concat_order[0] * S, concat_order[1] * S, concat_order[2] * S, concat_order[3] * S], axis=0)
    return S_full






if __name__=="__main__":
    #Sy1 = AssembleSx(2, 2, 16, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    #Sy2 = AssembleSSymm(2, 2, 16, "x", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    Sy1 = AssembleSySurf(2, 2, 10, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    Sy2 = P_AssembleSSurfSymm(2, 2, 10, "y", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    print(np.linalg.norm(Sy1.T@Sy1-Sy2.T@Sy2))



    






