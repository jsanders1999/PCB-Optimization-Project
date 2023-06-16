#### LEGACY CODE, ONLY USE IF THE PARRALEL ONE IS NOT WORKING ###


import numpy as np
import scipy.integrate as si
from numba import njit, jit
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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


#### The system matrices (steam functions coeffficients |--> Magnetic field at grid points in target volume ) ####
def AssembleSx(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    X = X.reshape((int(resolution**3)), order = "C") #order the volume lexographically
    Y = Y.reshape((int(resolution**3)), order = "C") #order the volume lexographically
    Z = Z.reshape((int(resolution**3)), order = "C") #order the volume lexographically


    Sx = np.zeros((resolution**3, NM))
    for i in tqdm(range(resolution**3), "Assembling Sx"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sx[i,j] = Bx_harm_Ztar(X[i], Y[i], Z[i], n, m)
    return Sx


def AssembleSy(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    X = X.reshape((int(resolution**3)), order = "C") #order the volume lexographically
    Y = Y.reshape((int(resolution**3)), order = "C") #order the volume lexographically
    Z = Z.reshape((int(resolution**3)), order = "C") #order the volume lexographically

    Sy = np.zeros((resolution**3, NM))
    for i in tqdm(range(resolution**3), "Assembling Sy"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sy[i,j] = By_harm_Ztar(X[i], Y[i], Z[i], n, m)
    return Sy

def AssembleSz(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    X = X.reshape((int(resolution**3)), order = "C") #order the volume lexographically
    Y = Y.reshape((int(resolution**3)), order = "C") #order the volume lexographically
    Z = Z.reshape((int(resolution**3)), order = "C") #order the volume lexographically

    Sz = np.zeros((resolution**3, NM))
    for i in tqdm(range(resolution**3), "Assembling Sz"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sz[i,j] = Bz_harm_Ztar(X[i], Y[i], Z[i], n, m)
    return Sz

#### The system matrices (steam functions coeffficients |--> Magnetic field at grid points in target volume ) ####
def AssembleSxSymm(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution)[:int(resolution/2)] #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)[:int(resolution/2)]
    z = np.linspace(V[4], V[5], resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically
    Y = Y.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically
    Z = Z.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically

    Sx = np.zeros((int(resolution**3/4), NM))
    for i in tqdm(range(int(resolution**3/4)), "Assembling Sx"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sx[i,j] = Bx_harm_Ztar(X[i], Y[i], Z[i], n, m)
    Sx_full = np.concatenate([Sx, -Sx, Sx, -Sx], axis = 0)
    return Sx_full

def AssembleSySymm(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution)[:int(resolution/2)] #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)[:int(resolution/2)]
    z = np.linspace(V[4], V[5], resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically
    Y = Y.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically
    Z = Z.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically

    Sy = np.zeros((int(resolution**3/4), NM))
    for i in tqdm(range(int(resolution**3/4)), "Assembling Sy"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sy[i,j] = By_harm_Ztar(X[i], Y[i], Z[i], n, m)
    Sy_full = np.concatenate([Sy, Sy, -Sy, -Sy], axis = 0)
    return Sy_full

def AssembleSzSymm(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution)[:int(resolution/2)] #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)[:int(resolution/2)]
    z = np.linspace(V[4], V[5], resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically
    Y = Y.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically
    Z = Z.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically

    Sz = np.zeros((int(resolution**3/4), NM))
    for i in tqdm(range(int(resolution**3/4)), "Assembling Sz"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sz[i,j] = Bz_harm_Ztar(X[i], Y[i], Z[i], n, m)
    Sz_full = np.concatenate([Sz, Sz, Sz, Sz], axis = 0)
    return Sz_full

#### The system matrices for the outer surface of the target volume (steam functions coeffficients |--> Magnetic field at grid points in target volume ) ####

def AssembleSxSurf(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape((resolution**3), order = "C") #order the volume lexographically
    Y = Y.reshape((resolution**3), order = "C") #order the volume lexographically
    Z = Z.reshape((resolution**3), order = "C") #order the volume lexographically
    
    #remove the interior elements of the target volume
    deletelist = []
    for i in range(1,resolution-1):
        for j in range(1,resolution-1):
            for k in range(1,resolution-1):
                deletelist += [resolution**2*i+resolution*j+k]
    X_new = np.delete(X, deletelist)
    Y_new = np.delete(Y, deletelist)
    Z_new = np.delete(Z, deletelist)

    Sx = np.zeros((resolution**3 -(resolution-2)**3, NM))
    for i in tqdm(range(len(X_new)), "Assembling Sx"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sx[i,j] = Bx_harm_Ztar(X_new[i], Y_new[i], Z_new[i], n, m)
    return Sx

def AssembleSySurf(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape((resolution**3), order = "C") #order the volume lexographically
    Y = Y.reshape((resolution**3), order = "C") #order the volume lexographically
    Z = Z.reshape((resolution**3), order = "C") #order the volume lexographically

    #remove the interior elements of the target volume
    deletelist = []
    for i in range(1,resolution-1):
        for j in range(1,resolution-1):
            for k in range(1,resolution-1):
                deletelist += [resolution**2*i+resolution*j+k]
    X_new = np.delete(X, deletelist)
    Y_new = np.delete(Y, deletelist)
    Z_new = np.delete(Z, deletelist)

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(X_new, Y_new, Z_new)
    #plt.show()

    Sy = np.zeros((resolution**3 -(resolution-2)**3, NM))
    for i in tqdm(range(len(X_new)), "Assembling Sy"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sy[i,j] = By_harm_Ztar(X_new[i], Y_new[i], Z_new[i], n, m)
    return Sy

def AssembleSzSurf(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape((resolution**3), order = "C") #order the volume lexographically
    Y = Y.reshape((resolution**3), order = "C") #order the volume lexographically
    Z = Z.reshape((resolution**3), order = "C") #order the volume lexographically

    #remove the interior elements of the target volume
    deletelist = []
    for i in range(1,resolution-1):
        for j in range(1,resolution-1):
            for k in range(1,resolution-1):
                deletelist += [resolution**2*i+resolution*j+k]
    X_new = np.delete(X, deletelist)
    Y_new = np.delete(Y, deletelist)
    Z_new = np.delete(Z, deletelist)

    Sz = np.zeros((resolution**3 -(resolution-2)**3, NM))
    for i in tqdm(range(len(X_new)), "Assembling Sz"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sz[i,j] = Bz_harm_Ztar(X_new[i], Y_new[i], Z_new[i], n, m)
    return Sz

def AssembleSxSurfZSym(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    if resolution//2 != resolution/2:
        print("Resolution is not even!")
        raise ValueError
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x)
    X = X.reshape((resolution**3), order = "C") #order the volume lexographically
    Y = Y.reshape((resolution**3), order = "C") #order the volume lexographically
    Z = Z.reshape((resolution**3), order = "C") #order the volume lexographically
    
    #remove the interior elements of the target volume
    deletelist = []
    for k in range(resolution):
        for j in range(resolution):
            for i in range(resolution):
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

    Sx = np.zeros((len(X_new), NM))
    for i in tqdm(range(len(X_new)), "Assembling Sx"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sx[i,j] = Bx_harm_Ztar(X_new[i], Y_new[i], Z_new[i], n, m)

    Sx_full = np.concatenate([Sx,-Sx, Sx, -Sx], axis = 0) #np.zeros((resolution**3-(resolution-2)**3, NM))
    #The ordering of the grid points in sx_full does not matter for the optimization!
    #Sx_full[0:(resolution**3-(resolution-2)**3)/4] = Sx
    #Sx_full[(resolution**3-(resolution-2)**3)/4:2*(resolution**3-(resolution-2)**3)/4] = -Sx
    #Sx_full[0:(resolution**3-(resolution-2)**3)/4] = Sx
    #Sx_full[0:(resolution**3-(resolution-2)**3)/4] = -Sx
                
    return Sx_full

def AssembleSySurfZSym(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    if resolution//2 != resolution/2:
        print("Resolution is not even!")
        raise ValueError
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x)
    X = X.reshape((resolution**3), order = "C") #order the volume lexographically
    Y = Y.reshape((resolution**3), order = "C") #order the volume lexographically
    Z = Z.reshape((resolution**3), order = "C") #order the volume lexographically
    
    #remove the interior elements of the target volume
    deletelist = []
    for k in range(resolution):
        for j in range(resolution):
            for i in range(resolution):
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

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(X_new, Y_new, Z_new)
    #plt.show()

    Sy = np.zeros((len(X_new), NM))
    for i in tqdm(range(len(X_new)), "Assembling Sy"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sy[i,j] = By_harm_Ztar(X_new[i], Y_new[i], Z_new[i], n, m)

    Sy_full = np.concatenate([Sy,Sy, -Sy, -Sy], axis = 0)#Sy_full = np.zeros((resolution**3-(resolution-2)**3, NM))
    #The ordering of the grid points in sx_full does not matter for the optimization!
    #Sy_full[0:(resolution**3-(resoltion-2)**3)/4] = Sy
    #Sy_full[0:(resolution**3-(resoltion-2)**3)/4] = -Sy
    #Sy_full[0:(resolution**3-(resoltion-2)**3)/4] = Sy
    #Sy_full[0:(resolution**3-(resoltion-2)**3)/4] = -Sy
                
    return Sy_full

def AssembleSzSurfZSym(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    if resolution//2 != resolution/2:
        print("Resolution is not even!")
        raise ValueError
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x)
    X = X.reshape((resolution**3), order = "C") #order the volume lexographically
    Y = Y.reshape((resolution**3), order = "C") #order the volume lexographically
    Z = Z.reshape((resolution**3), order = "C") #order the volume lexographically
    
    #remove the interior elements of the target volume
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

    Sz = np.zeros((len(X_new), NM))
    for i in tqdm(range(len(X_new)), "Assembling Sz"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sz[i,j] = Bz_harm_Ztar(X_new[i], Y_new[i], Z_new[i], n, m)

    Sz_full = np.concatenate([Sz, Sz, Sz, Sz], axis = 0)#Sz_full = np.zeros((resolution**3-(resolution-2)**3, NM))
    #The ordering of the grid points in sx_full does not matter for the optimization!
    #Sz_full[0:(resolution**3-(resoltion-2)**3)/4] = Sz
    #Sz_full[0:(resolution**3-(resoltion-2)**3)/4] = Sz
    #Sz_full[0:(resolution**3-(resoltion-2)**3)/4] = Sz
    #Sz_full[0:(resolution**3-(resoltion-2)**3)/4] = Sz
                
    return Sz_full


def AssembleSxSymBetter(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    X = X.reshape((int(resolution**3)), order = "C") #order the volume lexographically
    Y = Y.reshape((int(resolution**3)), order = "C") #order the volume lexographically
    Z = Z.reshape((int(resolution**3)), order = "C") #order the volume lexographically

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
                    temp = Bx_harm(X[i1], Y[i1], Z[i1], n, m)
                    S[i1,j] = temp
                    S[i2,j] = temp
                    S[i3,j] = -temp
                    S[i4,j] = -temp
    return S

def AssembleSySymBetter(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    X = X.reshape((int(resolution**3)), order = "C") #order the volume lexographically
    Y = Y.reshape((int(resolution**3)), order = "C") #order the volume lexographically
    Z = Z.reshape((int(resolution**3)), order = "C") #order the volume lexographically

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
                    temp = By_harm_Ztar(X[i1], Y[i1], Z[i1], n, m)
                    S[i1,j] = temp
                    S[i2,j] = -temp
                    S[i3,j] = -temp
                    S[i4,j] = temp
    return S

def AssembleSzSymBetter(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    X = X.reshape((int(resolution**3)), order = "C") #order the volume lexographically
    Y = Y.reshape((int(resolution**3)), order = "C") #order the volume lexographically
    Z = Z.reshape((int(resolution**3)), order = "C") #order the volume lexographically

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
                    temp = Bz_harm_Ztar(X[i1], Y[i1], Z[i1], n, m)
                    S[i1,j] = temp
                    S[i2,j] = temp
                    S[i3,j] = temp
                    S[i4,j] = temp
    return S

def AssembleSzSymmBetterBetter(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution)[:int(resolution/2)] #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)[:int(resolution/2)]
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    X = X.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically
    Y = Y.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically
    Z = Z.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically

    S_temp = np.zeros((int(resolution**3/4), NM))
    for i in tqdm(range(int(resolution**3/4)), "Assembling Sz"):
        for j in range(NM):
            m = j//N
            n = j%N
            S_temp[i,j] = Bz_harm_Ztar(X[i], Y[i], Z[i], n, m)

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
                    S[i1,j] = temp
                    S[i2,j] = temp
                    S[i3,j] = temp
                    S[i4,j] = temp

    return S

def P_AssembleSzSymmBetterBetter(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution)[:int(resolution/2)] #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)[:int(resolution/2)]
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    X = X.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically
    Y = Y.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically
    Z = Z.reshape((int(resolution**3/4)), order = "C") #order the volume lexographically

    def compute_s(i):
        temp_s = np.zeros(NM)
        for j in range(NM):
            m = j // N
            n = j % N
            temp_s[j] = Bz_harm_Ztar(X[i], Y[i], Z[i], n, m)
        return temp_s

    S_temp = np.array(
        Parallel(n_jobs=-1)(
            delayed(compute_s)(i) for i in tqdm(range(int(resolution ** 3 / 4)), f"Assembling Sz for the Full Volume using symmetry")
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
                    S[i1,j] = temp
                    S[i2,j] = temp
                    S[i3,j] = temp
                    S[i4,j] = temp

    return S




if __name__=="__main__":
    #AssembleSySurf(2, 2, 4, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    S1 = AssembleSz(4, 4, 8, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    S2 = P_AssembleSzSymmBetterBetter(4, 4, 8, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    print(np.linalg.norm(S1.T@S1-S2.T@S2))
    print(S1-S2)



