import numpy as np
from numba import njit
import matplotlib.pyplot as plt


cx = 0 #0 for z, 1 for x
#HalfPI = np.pi/2
pow = 1

#### Z as target direction ####
@njit
def HarmStreamFunc_Ztar(x,y,n,m):
    kx = (n+1/2)*np.pi
    ky = (m+1/2)*np.pi
    return np.sin( kx*(x**pow+1) ) * np.sin( ky*(y**pow+1) )/(np.sqrt(kx**2+ky**2))  #*(-1)**((n+1)//2+(m+1)//2)*

@njit
def delx_HarmStreamFunc_Ztar(x,y,n,m):
    kx = (n+1/2)*np.pi
    ky = (m+1/2)*np.pi
    return np.cos( kx*(x**pow+1) ) * np.sin( ky*(y**pow+1) ) * kx/(np.sqrt(kx**2+ky**2))

@njit
def dely_HarmStreamFunc_Ztar(x,y,n,m):
    kx = (n+1/2)*np.pi
    ky = (m+1/2)*np.pi
    return np.sin( kx*(x**pow+1) ) * np.cos( ky*(y**pow+1) ) * ky/(np.sqrt(kx**2+ky**2))


#### X as target direction ####
@njit
def HarmStreamFunc_Xtar(x,y,n,m):
    kx = (n+1)*np.pi
    ky = (m+1/2)*np.pi
    return np.sin( kx*(x**pow+1) ) * np.sin( ky*(y**pow+1) )/(np.sqrt(kx**2+ky**2))  #*(-1)**((n+1)//2+(m+1)//2)*

@njit
def delx_HarmStreamFunc_Xtar(x,y,n,m):
    kx = (n+1)*np.pi
    ky = (m+1/2)*np.pi
    return np.cos( kx*(x**pow+1) ) * np.sin( ky*(y**pow+1) ) * kx/(np.sqrt(kx**2+ky**2))

@njit
def dely_HarmStreamFunc_Xtar(x,y,n,m):
    kx = (n+1)*np.pi
    ky = (m+1/2)*np.pi
    return np.sin( kx*(x**pow+1) ) * np.cos( ky*(y**pow+1) ) * ky/(np.sqrt(kx**2+ky**2))


#### 1/r functions ####
@njit
def psi_x(x, y, z):
    return x/np.sqrt(x**2+y**2+z**2)**3

@njit
def psi_y(x, y, z):
    return y/np.sqrt(x**2+y**2+z**2)**3

@njit
def psi_z(x, y, z):
    return z/np.sqrt(x**2+y**2+z**2)**3

#         z
#         |
#         |
#         |
#         |
#         |
#---------+--------- x
#   |     |     |
#   \     |     /
#    \    |    /
#      ---|---
#         |
#         |
@njit
def x_par(s):
    return -np.cos(np.pi*s)/np.pi

@njit
def dxds_par(s):
    return 2*np.sin(np.pi*s)

@njit
def z_par(s):
    return np.sin(np.pi*s)/np.pi

@njit
def dzds_par(s):
    return np.cos(np.pi*s)

@njit
def K_x(s,t):
    return dxds_par(s,t)*dely_HarmStreamFunc(s,t)/np.sqrt(dxds_par(s,t)**2 + dzds_par(s,t)**2)

@njit
def K_y(s,t):
    return delx_HarmStreamFunc(s,t)

@njit
def K_z(s,t):
    return dzds_par(s,t)*dely_HarmStreamFunc(s,t)/np.sqrt(dxds_par(s,t)**2 + dzds_par(s,t)**2)

if __name__ == "__main__":
    fig, ax = plt.subplots(1,1)
    x = np.linspace(-1,1, 1000) 
    y = np.linspace(-1,1, 1000)
    X, Y = np.meshgrid(x, y)
    Sol = HarmStreamFunc_Xtar(X, Y, 0, 0)
    pc = ax.pcolormesh(X, Y, Sol, vmin = - max(np.max(Sol), np.min(Sol)), vmax = max(np.max(Sol), np.min(Sol)), )
    ax.set_aspect('equal')
    fig.colorbar(pc)

    x = np.linspace(-1,1, 10) 
    y = np.linspace(-1,1, 10)
    X, Y = np.meshgrid(x, y)
    Ux = dely_HarmStreamFunc_Xtar(X, Y, 0, 0)
    Uy = -delx_HarmStreamFunc_Xtar(X, Y, 0, 0)
    ax.quiver(X, Y, Ux, Uy)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.show()

    plt.style.use("seaborn-v0_8-dark")

    fig, ax = plt.subplots(1,1)
    x = np.linspace(-1,1, 1000)
    for n in [0,1,2]:
        y = HarmStreamFunc_Ztar(x,0,n,0)
        ax.plot(x,y)

    for n in [0,1,2]:
        y = HarmStreamFunc_Xtar(x,0,n,0)
        ax.plot(x,y)
    
    ax.set_xlabel("x [m]")
    
    plt.show()


