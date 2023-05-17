import numpy as np
from numba import njit


cx = 0 #0 for z, 1 for x
#HalfPI = np.pi/2


@njit
def HarmStreamFunc(x,y,n,m):
    kx = (n+(1+cx)/2)*np.pi
    ky = (m+1/2)*np.pi
    return np.sin( kx*(x+1) ) * np.sin( ky*(y+1) )/(np.sqrt(kx**2+ky**2))  #*(-1)**((n+1)//2+(m+1)//2)*

@njit
def delx_HarmStreamFunc(x,y,n,m):
    kx = (n+(1+cx)/2)*np.pi
    ky = (m+1/2)*np.pi
    return np.cos( kx*(x+1) ) * np.sin( ky*(y+1) ) * kx/(np.sqrt(kx**2+ky**2))

@njit
def dely_HarmStreamFunc(x,y,n,m):
    kx = (n+(1+cx)/2)*np.pi
    ky = (m+1/2)*np.pi
    return np.sin( kx*(x+1) ) * np.cos( ky*(y+1) ) * ky/(np.sqrt(kx**2+ky**2))

@njit
def psi_x(x, y, z):
    return x/np.sqrt(x**2+y**2+z**2)**3

@njit
def psi_y(x, y, z):
    return y/np.sqrt(x**2+y**2+z**2)**3

@njit
def psi_z(x, y, z):
    return z/np.sqrt(x**2+y**2+z**2)**3


