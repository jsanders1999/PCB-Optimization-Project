import numpy as np
import scipy.integrate as si
from numba import njit, jit
from tqdm import tqdm
import matplotlib.pyplot as plt
#from joblib import Parallel, delayed

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

def Test_Conv():
    print(psiz_conv_dely_harm(0.5, 0.5, 0.5, 0, 0,))
    print(psiz_conv_delx_harm(1.0, 1.5, 2.5, 0, 0,))
    print(psixy_conv_delxy_harm(0.5, 0.5, 0.5, 0, 0))
    return


#### The system matrices (steam functions coeffficients |--> Magnetic field at grid points in target volume ) ####
def AssembleSx(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape((resolution**3), order = "C") #order the volume lexographically
    Y = Y.reshape((resolution**3), order = "C") #order the volume lexographically
    Z = Z.reshape((resolution**3), order = "C") #order the volume lexographically

    Sx = np.zeros((resolution**3, NM))
    for i in tqdm(range(resolution**3), "Assembling Sx"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sx[i,j] = psiz_conv_delx_harm(X[i], Y[i], Z[i], n, m)
    return Sx

def AssembleSy(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape((resolution**3), order = "C") #order the volume lexographically
    Y = Y.reshape((resolution**3), order = "C") #order the volume lexographically
    Z = Z.reshape((resolution**3), order = "C") #order the volume lexographically

    Sy = np.zeros((resolution**3, NM))
    for i in tqdm(range(resolution**3), "Assembling Sy"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sy[i,j] = psiz_conv_dely_harm(X[i], Y[i], Z[i], n, m)
    return Sy

def AssembleSz(N, M, resolution, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1]):
    NM = N*M
    x = np.linspace(V[0], V[1], resolution) #Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape((resolution**3), order = "C") #order the volume lexographically
    Y = Y.reshape((resolution**3), order = "C") #order the volume lexographically
    Z = Z.reshape((resolution**3), order = "C") #order the volume lexographically

    Sz = np.zeros((resolution**3, NM))
    for i in tqdm(range(resolution**3), "Assembling Sz"):
        for j in range(NM):
            m = j//N
            n = j%N
            Sz[i,j] = psixy_conv_delxy_harm(X[i], Y[i], Z[i], n, m)
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
            Sx[i,j] = psiz_conv_delx_harm(X[i], Y[i], Z[i], n, m)
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
            Sy[i,j] = psiz_conv_dely_harm(X[i], Y[i], Z[i], n, m)
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
            Sz[i,j] = psixy_conv_delxy_harm(X[i], Y[i], Z[i], n, m)
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
            Sx[i,j] = psiz_conv_delx_harm(X_new[i], Y_new[i], Z_new[i], n, m)
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
            Sy[i,j] = psiz_conv_dely_harm(X_new[i], Y_new[i], Z_new[i], n, m)
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
            Sz[i,j] = psixy_conv_delxy_harm(X_new[i], Y_new[i], Z_new[i], n, m)
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
            Sx[i,j] = psiz_conv_delx_harm(X_new[i], Y_new[i], Z_new[i], n, m)

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
            Sy[i,j] = psiz_conv_dely_harm(X_new[i], Y_new[i], Z_new[i], n, m)

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
            Sz[i,j] = psixy_conv_delxy_harm(X_new[i], Y_new[i], Z_new[i], n, m)

    Sz_full = np.concatenate([Sz, Sz, Sz, Sz], axis = 0)#Sz_full = np.zeros((resolution**3-(resolution-2)**3, NM))
    #The ordering of the grid points in sx_full does not matter for the optimization!
    #Sz_full[0:(resolution**3-(resoltion-2)**3)/4] = Sz
    #Sz_full[0:(resolution**3-(resoltion-2)**3)/4] = Sz
    #Sz_full[0:(resolution**3-(resoltion-2)**3)/4] = Sz
    #Sz_full[0:(resolution**3-(resoltion-2)**3)/4] = Sz
                
    return Sz_full

if __name__=="__main__":
    #AssembleSySurf(2, 2, 4, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    Sy1 = AssembleSx(2, 2, 6, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    Sy2 = AssembleSxSymm(2, 2, 6, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5,  -0.5, 0.5,  0.1, 1.1])
    print(np.linalg.norm(Sy1.T@Sy1-Sy2.T@Sy2))



    











#### The Q inner product ####
#def Qprod_integrand(x, y, z, p, q, n, m): #domain2d = [-1, 1, -1, 1]):
#    return psiz_conv_dely_harm(x, y, z, p, q)*psiz_conv_dely_harm(x, y, z, n, m)\
#         + psiz_conv_delx_harm(x, y, z, p, q)*psiz_conv_delx_harm(x, y, z, n, m)\
#         + psixy_conv_delxy_harm(x, y, z, p, q)*psixy_conv_delxy_harm(x, y, z, n, m)
#Qprod_integrand = np.vectorize(Qprod_integrand)

#def Qprod(p, q, n, m, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
#    # int(int(int(  (-psi_z conv dely_stream)**2 + (-psi_z conv delx_stream )**2  + (  )**2 )))
#    return si.tplquad(Qprod_integrand, *V, args = (p, q, n, m))[0] 

# def Qprod_num(p, q, n, m, Omega = [-1.0, 1.0, -1.0, 1.0], V = [-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
#     res = 8
#     x = np.linspace(-0.5,0.5,res)
#   
#   y = np.linspace(-0.5,0.5,res)
#     z = np.linspace(0.1,1.1,res)
#     X, Y, Z = np.meshgrid(x, y, z)
#     # int(int(int(  (-psi_z conv dely_stream)**2 + (-psi_z conv delx_stream )**2  + (  )**2 )))
#     return np.sum(Qprod_integrand(X,Y,Z, p, q, n, m))/res**3


#### The R inner product ####
#def Rprod(Omega = [[-1.0, 1.0], [-1.0, 1.0]], V = [[-0.5, 0.5], [-0.5, 0.5], [0.1, 1.1]]):
#    return

#### The P inner product ####
#def Pprod(Omega = [[-1.0, 1.0], [-1.0, 1.0]]):
#    #NOTE: CAN BE DONE ANANLYTICALLY
#    return

#def AssembleQ(NM, Omega = [[-1.0, 1.0], [-1.0, 1.0]], V = [[-0.5, 0.5], [-0.5, 0.5], [0.1, 1.1]]):
#    Q = np.zeros((NM, NM))
#    for i in range(NM):
#        for j in range(i,NM):
#            n = 0 ## make i lexographic
#            m = 0 ## make j lexographic
#            Q[i,j] = Qprod(n,m, Omega, V)
#            Q[j,i] = Q[i,j]

