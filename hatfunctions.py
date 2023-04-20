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
