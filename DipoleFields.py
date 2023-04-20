import numpy as np
from numba import njit


err = 0.1
@njit
def B_dipole_x(x,y,z, err = 1e-5):
    # np.where to eliminate case of close to origin
    return np.where(x**2+y**2+z**2 < err, 0, 4/3*x*z/(x**2+y**2+z**2)**(7/2))

@njit    
def B_dipole_y(x,y,z, err = 1e-5):
    return np.where(x**2+y**2+z**2 < err, 0, 4/3*y*z/(x**2+y**2+z**2)**(7/2))

@njit
def B_dipole_z(x,y,z, err = 1e-5):
    return np.where(x**2+y**2+z**2 < err, 0, 4*(z**2/(x**2+y**2+z**2) - 1/3)/(x**2+y**2+z**2)**(3/2))

@njit
def B_multi_x(x,y,z, err = 1e-5):
    m_alpha = 1/2/4/5
    return np.where(x**2+y**2+z**2 < err, 0, 3 * m_alpha * z * (4 * x ** 2 + 5 * x * y - y ** 2 - z ** 2) * (x ** 2 + y ** 2 + z ** 2) ** (-7 / 2))

@njit
def B_multi_y(x,y,z, err = 1e-5):
    m_alpha = 1/2/4/5
    return np.where(x**2+y**2+z**2 < err, 0, -3 * m_alpha * z * (x ** 2 - 5 * x * y - 4 * y ** 2 + z ** 2) * (x ** 2 + y ** 2 + z ** 2) ** (-7 / 2))

@njit
def B_multi_z(x,y,z, err = 1e-5):
    m_alpha = 1/2/4/5
    return np.where(x**2+y**2+z**2 < err, 0, -3 * m_alpha * (x + y) * (x ** 2 + y ** 2 - 4 * z ** 2) * (x ** 2 + y ** 2 + z ** 2) ** (-7 / 2))

