# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
from numba import njit


@njit
def norm_impl(x):
    squared = x**2
    summed = np.sum(squared, axis = 0)
    return np.sqrt(summed)

@njit
def calc_mag_field_contour(x_i, x_f, polarity, x_target):
    """
    input: x_target array[3] = [x, y, z]
    """
    # norm of each line segment
    L = norm_impl(x_i - x_f)

    # transform x_target such that it can be broadcasted with x_i/f
    tmp = np.zeros((3,1))
    tmp[:,0]+= x_target
    x_target = tmp

    R_i = - x_i + x_target
    R_f = - x_f + x_target

    norm_R_i = norm_impl(R_i)
    norm_R_f = norm_impl(R_f)
    
    # numba does not support cross with axis argument so the arrays need to be transposed to circumvent that argument
    # the result is transposed again to give the correct result
    cross = np.cross((x_f - x_i).T, R_i.T)
    direction = cross.T/L

    epsilon = L/(norm_R_i + norm_R_f)
    factor = 2*epsilon/((1-epsilon**2)*norm_R_i*norm_R_f)

    field = direction*factor*polarity

    return np.sum(field, axis = 1)

@njit
def calc_mag_field_c_cube(x_i,x_f,polarity, X, Y, Z, field):
    num = X.shape[0]
    for i in range(num):
        for j in range(num):
            for k in range(num):
                x_target = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])

                field[i,j,k,:] += calc_mag_field_contour(x_i, x_f, polarity, x_target)
    

class Contour:
    def __init__(self, points, polarity):
        """
        input: points array[2,n]: [x0,...,xn] where x0 = xn
                                  [y0,...,yn] where y0 = yn 
                                  paths need to be drawn clockwise, to indicate a different current flow the
                                  polarity can be used
               polarity = 1 or -1
        """
        self.points = points
        self.n_segments = points.shape[1] - 1
        self.x_points = points[0,:]
        self.y_points = points[1,:]
        self.polarity = polarity
        
        self.x_i = np.vstack((self.points[:,:-1], np.zeros(self.n_segments)))
        self.x_f = np.vstack((self.points[:,1:], np.zeros(self.n_segments)))


    def calc_mag_field(self, x_target):
        return calc_mag_field_contour(self.x_i, self.x_f, self.polarity, x_target)
    

    def plot_contour(self, fig = None, ax = None):
        if (not fig) and (not ax):
            fig, ax = plt.subplots(1,1)
        
        if self.polarity == 1:
            style = 'b-' 
        else:
            style = 'r-'

        for i in range(self.n_segments):
            x_p = self.x_points[i:i+2]
            y_p = self.y_points[i:i+2]

            ax.plot(x_p, y_p, style)
        
        return fig, ax



class PCB_c:
    def __init__(self, contours, cube):
        """
        input: contours: list of contours
               cube: Cube 
        """
        self.contours = contours
        self.cube = cube
        self.calc_mag_field()

    def calc_mag_field(self):
        self.field = np.zeros((*self.cube.X.shape,3))

        # first we loop over the contours 
        for c in self.contours:
            # loop over points is done in separate function which is jitted
            calc_mag_field_c_cube(c.x_i, c.x_f, c.polarity, self.cube.X, self.cube.Y, self.cube.Z, self.field)
