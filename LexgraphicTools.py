import numpy as np
from numba import njit


@njit
def lex_to_cart_2D(I, N):
    i = I//N
    j = I - i*N
    return i, j

@njit
def cart_to_lex_2D(i, j, N):
    return j + i*N

@njit
def lex_to_cart_3D(I, N):
    k = I//(N**2)
    i = (I - k*N**2)//N
    j = I - i*N - k*N**2
    return i, j, k

@njit
def cart_to_lex_3D(i, j, k, N):
    return j + i*N + k*N**2

# function to reshape array: 1D to 3D
@njit
def reshape_array_lex_cart(arr, N):
    new = np.zeros((N,N,N))
    # place each element in the correct place of the 3D array
    for I, el in enumerate(arr):
        i,j,k = lex_to_cart_3D(I, N)
        new[i,j,k] = el
    return new


@njit
def reshape_array_lex_cart_sides(arr, N):
    new = np.zeros((6,N,N))
    # place each element in the correct place of the 3D array
    for I, el in enumerate(arr):
        i,j,k = lex_to_cart_3D(I, N)
        new[k,i,j] = el
    return new