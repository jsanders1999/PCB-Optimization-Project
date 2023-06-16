import numpy as np
import scipy.integrate as si
from numba import njit, jit
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from MatrixConstructors import AssembleSx, AssembleSy, AssembleSz, AssembleSySurf

from Streamfunctions import *

##### Integrands to find B in x, y, z #####

# dBx
@njit
def dBx_Ztar(s, t, x, y, z, n, m):
    return -psi_z(x-s, y-t, z)*delx_HarmStreamFunc_Ztar(s,t,n,m)

#dBy
@njit
def dBy_Ztar(s, t, x, y, z, n, m):
    return -psi_z(x-s, y-t, z)*dely_HarmStreamFunc_Ztar(s,t,n,m)

#dBz
@njit
def dBz_Ztar(s, t, x, y, z, n, m):
    return psi_x(x-s, y-t, z)*delx_HarmStreamFunc_Ztar(s,t,n,m) + psi_y(x-s, y-t, z)*dely_HarmStreamFunc_Ztar(s,t,n,m)

#### B in x, y, z ####
def Bx_harm_Ztar(x, y, z, n, m): #Bx
    return si.dblquad(dBx_Ztar, -1, 1, -1, 1, args = (x, y, z, n, m))[0]

def By_harm_Ztar(x, y, z, n, m): #By
    return si.dblquad(dBy_Ztar, -1, 1, -1, 1, args = (x, y, z, n, m))[0]

def Bz_harm_Ztar(x, y, z, n, m): #Bz
    return si.dblquad(dBz_Ztar, -1, 1, -1, 1, args = (x, y, z, n, m))[0]


# dBx
@njit
def dBx_Xtar(s, t, x, y, z, n, m):
    return -psi_z(x-s, y-t, z)*delx_HarmStreamFunc_Xtar(s,t,n,m)

#dBy
@njit
def dBy_Xtar(s, t, x, y, z, n, m):
    return -psi_z(x-s, y-t, z)*dely_HarmStreamFunc_Xtar(s,t,n,m)

#dBz
@njit
def dBz_Xtar(s, t, x, y, z, n, m):
    return psi_x(x-s, y-t, z)*delx_HarmStreamFunc_Xtar(s,t,n,m) + psi_y(x-s, y-t, z)*dely_HarmStreamFunc_Xtar(s,t,n,m)

#### B in x, y, z ####
def Bx_harm_Xtar(x, y, z, n, m): #Bx
    return si.dblquad(dBx_Xtar, -1, 1, -1, 1, args = (x, y, z, n, m))[0]

def By_harm_Xtar(x, y, z, n, m): #By
    return si.dblquad(dBy_Xtar, -1, 1, -1, 1, args = (x, y, z, n, m))[0]

def Bz_harm_Xtar(x, y, z, n, m): #Bz
    return si.dblquad(dBz_Xtar, -1, 1, -1, 1, args = (x, y, z, n, m))[0]



#### The system matrices (steam functions coeffficients |--> Magnetic field at grid points in target volume ) ####

def P_AssembleS(N, M, resolution, direction, tar_dir, Omega=[-1.0, 1.0, -1.0, 1.0], V=[-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
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
        if tar_dir == "x":
            B = Bx_harm_Xtar
        elif tar_dir == "z":
            B = Bx_harm_Ztar
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
    elif direction == "y":
        if tar_dir == "x":
            B = By_harm_Xtar
        elif tar_dir == "z":
            B = By_harm_Ztar
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
    elif direction == "z":
        if tar_dir == "x":
            B = Bz_harm_Xtar
        elif tar_dir == "z":
            B = Bz_harm_Ztar
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
    else:
        raise ValueError("Invalid direction. Must be 'x', 'y', or 'z'.")

    def compute_s(i):
        temp_s = np.zeros(NM)
        for j in range(NM):
            m = j // N
            n = j % N
            temp_s[j] = B(X[i], Y[i], Z[i], n, m)
        return temp_s

    S = np.array(
        Parallel(n_jobs=-1)(
            delayed(compute_s)(i) for i in tqdm(range(resolution ** 3), f"Assembling S{direction} for the Full Volume")
        )
    )

    return S


#### The system matrices (steam functions coeffficients |--> Magnetic field at grid points in target volume ) ####

def P_AssembleSSymm(N, M, resolution, direction, tar_dir,  Omega=[-1.0, 1.0, -1.0, 1.0], V=[-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
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
        if tar_dir == "x":
            B = Bx_harm_Xtar
            concat_order = [1, 1, 1, 1]
        elif tar_dir == "z":
            B = Bx_harm_Ztar
            concat_order = [1, 1, -1, -1]
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
    elif direction == "y":
        if tar_dir == "x":
            B = By_harm_Xtar
            concat_order = [1, -1, 1, -1]
        elif tar_dir == "z":
            B = By_harm_Ztar
            concat_order = [1, -1, -1, 1]
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
    elif direction == "z":
        if tar_dir == "x":
            B = Bz_harm_Xtar
            concat_order = [1, 1, -1, -1]
        elif tar_dir == "z":
            B = Bz_harm_Ztar
            concat_order = [1, 1, 1, 1]
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
    else:
        raise ValueError("Invalid direction. Must be 'x', 'y', or 'z'.")

    def compute_s(i):
        temp_s = np.zeros(NM)
        for j in range(NM):
            m = j // N
            n = j % N
            temp_s[j] = B(X[i], Y[i], Z[i], n, m)
        return temp_s

    S = np.array(
        Parallel(n_jobs=-1)(
            delayed(compute_s)(i) for i in tqdm(range(int(resolution ** 3 / 4)), f"Assembling S{direction} for the Full Volume using symmetry")
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
        if tar_dir == "x":
            B = Bx_harm_Xtar
        elif tar_dir == "z":
            B = Bx_harm_Ztar
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
    elif direction == "y":
        if tar_dir == "x":
            B = By_harm_Xtar
        elif tar_dir == "z":
            B = By_harm_Ztar
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
    elif direction == "z":
        if tar_dir == "x":
            B = Bz_harm_Xtar
        elif tar_dir == "z":
            B = Bz_harm_Ztar
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
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

    def compute_s(i):
        temp_s = np.zeros(NM)
        for j in range(NM):
            m = j // N
            n = j % N
            temp_s[j] = B(X_new[i], Y_new[i], Z_new[i], n, m)
        return temp_s

    S = np.array(
        Parallel(n_jobs=-1)(
            delayed(compute_s)(i) for i in tqdm(range((resolution**3 -(resolution-2)**3)), f"Assembling S{direction} for the edge of the Volume")
        )
    )

    return S

def P_AssembleSSurfSymm(N, M, resolution, direction, tar_dir, Omega=[-1.0, 1.0, -1.0, 1.0], V=[-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
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
        if tar_dir == "x":
            B = Bx_harm_Xtar
            concat_order = [1, 1, 1, 1]
        elif tar_dir == "z":
            B = Bx_harm_Ztar
            concat_order = [1, 1, -1, -1]
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
    elif direction == "y":
        if tar_dir == "x":
            B = By_harm_Xtar
            concat_order = [1, -1, 1, -1]
        elif tar_dir == "z":
            B = By_harm_Ztar
            concat_order = [1, -1, -1, 1]
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
    elif direction == "z":
        if tar_dir == "x":
            B = Bz_harm_Xtar
            concat_order = [1, 1, -1, -1]
        elif tar_dir == "z":
            B = Bz_harm_Ztar
            concat_order = [1, 1, 1, 1]
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
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

    def compute_s(i):
        temp_s = np.zeros(NM)
        for j in range(NM):
            m = j // N
            n = j % N
            temp_s[j] = B(X_new[i], Y_new[i], Z_new[i], n, m)
        return temp_s

    S = np.array(
        Parallel(n_jobs=-1)(
            delayed(compute_s)(i) for i in tqdm(range(int((resolution**3 -(resolution-2)**3)/4)), f"Assembling S{direction} for the edge of the Volume using Symmetry")
        )
    )

    S_full = np.concatenate([concat_order[0] * S, concat_order[1] * S, concat_order[2] * S, concat_order[3] * S], axis=0)
    return S_full


def P_AssembleSSymRightOrder(N, M, resolution, direction, tar_dir, Omega=[-1.0, 1.0, -1.0, 1.0], V=[-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
    if (V[0] + V[1]) / 2 != (Omega[0] + Omega[1]) / 2 or (V[2] + V[3]) / 2 != (Omega[2] + Omega[3]) / 2:
        raise ValueError("The box is not in the middle of the surface.")
    if resolution%2 != 0:
        raise ValueError("The resultion of {} is not even".format(resolution))
    NM = N*M
    x = np.linspace(V[0], V[1], resolution)[:int(resolution/2)] #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)[:int(resolution/2)]
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    X = X.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically
    Y = Y.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically
    Z = Z.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically

    if direction == "x":
        if tar_dir == "x":
            B = Bx_harm_Xtar
            concat_order = [1, 1, 1, 1]
        elif tar_dir == "z":
            B = Bx_harm_Ztar
            concat_order = [1, 1, -1, -1]
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
    elif direction == "y":
        if tar_dir == "x":
            B = By_harm_Xtar
            concat_order = [1, -1, 1, -1]
        elif tar_dir == "z":
            B = By_harm_Ztar
            concat_order = [1, -1, -1, 1]
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
    elif direction == "z":
        if tar_dir == "x":
            B = Bz_harm_Xtar
            concat_order = [1, 1, -1, -1]
        elif tar_dir == "z":
            B = Bz_harm_Ztar
            concat_order = [1, 1, 1, 1]
        else:
            raise ValueError("Invalid target direction. Must be 'x' or 'z'.")
    else:
        raise ValueError("Invalid direction. Must be 'x', 'y', or 'z'.")

    def compute_s(i):
        temp_s = np.zeros(NM)
        for j in range(NM):
            m = j // N
            n = j % N
            temp_s[j] = B(X[i], Y[i], Z[i], n, m)
        return temp_s

    S_temp = np.array(
        Parallel(n_jobs=-1)(
            delayed(compute_s)(i) for i in tqdm(range(int(resolution ** 3 / 4)), f"Assembling S{direction} for the Full Volume using symmetry")
        )
    )

    S = np.zeros((resolution**3, NM))
    
    for iz in range(resolution):
        for iy in range(int(resolution/2)):
            for ix in range(int(resolution/2)):
                i1 = iz*resolution**2 + iy*resolution + ix  #-x, -y quadrant
                i2 = iz*resolution**2 + (resolution-1-iy)*resolution + ix #-x, +y quandrant
                i3 = iz*resolution**2 + (resolution-1-iy)*resolution + (resolution-1-ix) #+x, +y quadrant
                i4 = iz*resolution**2 + iy*resolution + (resolution-1-ix) #+x, -y quandrant
        
                for j in range(NM):
                    
                    m = j//N #Harmonic number i y dir
                    n = j%N #Harmonic number in x dir
                    
                    #Calculate the integral for the first quadrant
                    temp = S_temp[iz*(int(resolution/2))**2 + iy*int(resolution/2) + ix, j]
                    S[i1,j] = concat_order[0]*temp
                    S[i2,j] = concat_order[1]*temp
                    S[i3,j] = concat_order[2]*temp
                    S[i4,j] = concat_order[3]*temp

    return S






if __name__=="__main__":
    #Sy1 = AssembleSx(2, 2, 16, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    #Sy2 = AssembleSSymm(2, 2, 16, "x", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    S1 = AssembleSz(2, 2, 8, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    #S2 = P_AssembleSSurfSymm(2, 2, 10, "y", Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    S3 = P_AssembleSSymRightOrder(2, 2, 8, "z", "z")
    print(np.linalg.norm(S1.T@S1-S3.T@S3))
    print(S1-S3)



"""
def P_AssembleSSymmXtar(N, M, resolution, direction, Omega=[-1.0, 1.0, -1.0, 1.0], V=[-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
    
    TODO: Complete this function
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
        B = Bx_harm_Ztar
        concat_order = [1, 1, 1, 1]
    elif direction == "y":
        B = By_harm_Ztar
        concat_order = [1, -1, 1, -1]
    elif direction == "z":
        B = Bz_harm_Ztar
        concat_order = [1, 1, -1, -1]
    else:
        raise ValueError("Invalid direction. Must be 'x', 'y', or 'z'.")

    def compute_s(i):
        temp_s = np.zeros(NM)
        for j in range(NM):
            m = j // N
            n = j % N
            temp_s[j] = B(X[i], Y[i], Z[i], n, m)
        return temp_s

    S = np.array(
        Parallel(n_jobs=-1)(
            delayed(compute_s)(i) for i in tqdm(range(int(resolution ** 3 / 4)), f"Assembling S{direction} for the Full Volume using symmetry")
        )
    )

    S_full = np.concatenate([concat_order[0] * S, concat_order[1] * S, concat_order[2] * S, concat_order[3] * S], axis=0)
    return S_full
"""


    






