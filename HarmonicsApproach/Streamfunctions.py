import numpy as np
from numba import njit


cx = 0 #0 for z, 1 for x


@njit
def HarmStreamFunc(x,y,n,m):
    return 2/(np.pi*np.sqrt((2*n+1+cx)**2+(2*m+1)**2))*np.sin( (2*n+1+cx)*np.pi/2*(x+1) ) * np.sin( (2*m+1)*np.pi/2*(y+1) )  #*(-1)**((n+1)//2+(m+1)//2)*

@njit
def delx_HarmStreamFunc(x,y,n,m):
    return 2/(np.pi*np.sqrt((2*n+1+cx)**2+(2*m+1)**2))*(2*n+1+cx)*np.pi/2*np.cos( (2*n+1+cx)*np.pi/2*(x+1) ) * np.sin( (2*m+1)*np.pi/2*(y+1) )

@njit
def dely_HarmStreamFunc(x,y,n,m):
    return 2/(np.pi*np.sqrt((2*n+1+cx)**2+(2*m+1)**2))*(2*m+1)*np.pi/2*np.sin( (2*n+1+cx)*np.pi/2*(x+1) ) * np.cos( (2*m+1)*np.pi/2*(y+1) )

@njit
def psi_x(x, y, z):
    return x/np.sqrt(x**2+y**2+z**2)**3

@njit
def psi_y(x, y, z):
    return y/np.sqrt(x**2+y**2+z**2)**3

@njit
def psi_z(x, y, z):
    return z/np.sqrt(x**2+y**2+z**2)**3
