import numpy as np
from numba import njit

@njit
def K0_x(x_s, y_s):
    domain14 = (np.absolute(y_s)>np.absolute(x_s))
    boxx = np.absolute(x_s)<=1
    boxy = np.absolute(y_s)<=1
    return -np.sign(y_s)*domain14*boxx*boxy#(y_s-np.sign(y_s))*domain14*boxx*boxy

@njit
def K0_y(x_s, y_s):
    domain23 = (np.absolute(y_s)<=np.absolute(x_s))
    boxx = np.absolute(x_s)<=1
    boxy = np.absolute(y_s)<=1
    return np.sign(x_s)*domain23*boxx*boxy#-(x_s-np.sign(x_s))*domain23*boxx*boxy

@njit
def curl_potential_0(x_s, y_s):
    domain14 = (np.absolute(y_s)>np.absolute(x_s))
    domain23 = (np.absolute(y_s)<=np.absolute(x_s))
    boxx = np.absolute(x_s)<=1
    boxy = np.absolute(y_s)<=1
    f_z14 = 1 - np.abs(y_s)
    f_z23 = 1 - np.abs(x_s)
    const = 0
    return f_z14*domain14*boxx*boxy + f_z23*domain23*boxx*boxy + const*boxx*boxy

@njit
def lambda_1D(x,x_0):
    return np.where(np.abs(x) <= x_0, 1-np.abs(x)/x_0, 0)

@njit
def dlambda_1D(x,x_0):
    res = np.where(np.abs(x) <= x_0, 1, 0)
    return np.where(x < 0, res/x_0, -res/x_0)

@njit
def lambda_2D(x, y, x_0):
    return lambda_1D(x,x_0)*lambda_1D(y,x_0)

@njit
def current_lambda(x, y, x_0):
    y_coeff = dlambda_1D(x, x_0) * lambda_1D(y, x_0)
    x_coeff = lambda_1D(x, x_0) * dlambda_1D(y, x_0)
    return x_coeff, -y_coeff
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    x_min = -2
    x_max = 2
    N = 10
    x_0 = 2
    
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(x_min, x_max, N)
    
    X, Y = np.meshgrid(x,y)
    
    curr_x, curr_y = current_lambda(X,Y,x_0)
    
    fig, ax = plt.subplots(1,1)
    
    # # plot 1D hat function
    # ax.plot(x, lambda_1D(x,x_0))
    
    # # plot 1D hat function derivative
    # ax.plot(x, dlambda_1D(x,x_0))
    
    # plot 2D hat function
    ax.pcolormesh(X,Y,lambda_2D(X, Y, x_0))
    
    # plot quiver of current
    ax.quiver(X, Y, curr_x, curr_y)
    