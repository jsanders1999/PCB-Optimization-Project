import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def meshgrid(x, y, z):
    xx = np.empty(shape=(x.size, y.size, z.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size, z.size), dtype=y.dtype)
    zz = np.empty(shape=(x.size, y.size, z.size), dtype=z.dtype)
    for i in range(z.size):
        for j in range(y.size):
            for k in range(x.size):
                xx[i,j,k] = x[k]  # change to x[k] if indexing xy
                yy[i,j,k] = y[j]  # change to y[j] if indexing xy
                zz[i,j,k] = z[i]  # change to z[i] if indexing xy
    return xx, yy, zz

@njit
def meshgrid(x, y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in range(y.size):
        for k in range(x.size):
            xx[j,k] = x[k]  # change to x[k] if indexing xy
            yy[j,k] = y[j]  # change to y[j] if indexing xy
    return xx, yy

@njit
def phi(x, y, n, m):
    return np.sin((n+1)*np.pi/2*(x+1))*np.sin((m+1)*np.pi/2*(y+1))

@njit
def K_x(x, y, n, m):
    return (m+1)*np.pi/2*np.sin((n+1)*np.pi/2*(x+1))*np.cos((m+1)*np.pi/2*(y+1))

@njit
def K_y(x, y, n, m):
    return -(n+1)*np.pi/2*np.cos((n+1)*np.pi/2*(x+1))*np.sin((m+1)*np.pi/2*(y+1))


#@njit
def Biot_Savart(k_x, k_y, n, m, x, y, z):
    #current(s,t)
    #Does a beun numeric integration with a sum
    RES = 100
    grid_x = np.linspace(-1, 1, RES)
    grid_y = np.linspace(-1, 1, RES)
    X_s, Y_s = meshgrid(grid_x, grid_y)
    dsdt = 1/RES**2
    r = np.zeros((3, *X_s.shape))#np.array([x-X_s, y-Y_s, z], dtype=object) #np.kron?
    r[0, :, :] = x-X_s
    r[1, :, :] = y-Y_s
    r[2, :, :] = z
    B_x = dsdt*np.sum( r[2, :, :]*k_y(X_s, Y_s , n, m) / np.linalg.norm(r, axis = 0)**3 ) #specify axis for speedup
    B_y = dsdt*np.sum( -r[2, :, :]*k_x(X_s, Y_s , n, m) / np.linalg.norm(r, axis = 0)**3 )
    B_z = dsdt*np.sum( (-r[0, :, :]*k_y(X_s, Y_s , n, m) + r[1, :, :]*k_x(X_s,Y_s, n, m) )/ np.linalg.norm(r, axis = 0)**3 )
    return np.array([B_x, B_y, B_z])

#@njit
def Assemble_S(N, M, x, y, z):
    S = np.zeros((3,N+1,M+1)) 
    for n in range(N+1):
        for m in range(M+1):
            S[:,n,m] = Biot_Savart(K_x, K_y, n, m, x, y, z)
    return S

def plot_S(n,m):
    X, Z = np.mgrid[-1:1:20j, -1:1:20j]
    y = 0
    B_X = np.ones(X.shape)
    B_Z = np.ones(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            res = Assemble_S(n, m, X[i,j], y, Z[i,j])
            B_X[i,j] = res[0,n,m]
            B_Z[i,j] = res[2,n,m]
    
    fig, ax = plt.subplots(1,1)


    ax.set_xlabel('x')
    ax.set_ylabel('z')

    #U_x = 0.25*self.field[:,:,:,0]/np.max(self.field)
    #U_y = 0.25*self.field[:,:,:,1]/np.max(self.field)
    #U_z = 0.25*self.field[:,:,:,2]/np.max(self.field)

    ax.quiver(X, Z, B_X, B_Z)
    return fig, ax


    
        

if __name__ == "__main__":
    #print(Assemble_S(1, 1, 0.1, 0.1, 0.1))
    plot_S(1,2)
    plt.show()
