# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
from numba import njit
from contourpy import contour_generator
from tqdm import tqdm

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
        
        if self.polarity == -1:
            style = 'b-' 
        else:
            style = 'r-'

        # for i in range(self.n_segments):
        #     x_p = self.x_points[i:i+2]
        #     y_p = self.y_points[i:i+2]

        ax.plot(self.x_points, self.y_points, style, linewidth = 0.5)
        
        # you can use an arrow to check if all the contours are plotted clockwise
        # ax.arrow(x_p[0], y_p[0], x_p[1] - x_p[0], y_p[1] - y_p[0], shape='full', lw=0, length_includes_head=True, head_width=.05)
        
        return fig, ax
 
 
def density_to_loops(U, num_levels, x, y):
    cont_gen = contour_generator(z=U, x = x, y = y, )
    u_min = np.min(U)
    u_max = np.max(U)
    
    levels = np.linspace(u_min, u_max, num_levels+2)[1:-1]
    
    # contours are drawn clockwise 
    contours = []
    
    for lvl in tqdm(levels, desc="looping over levels"):
        if lvl < 0:
            polarity = 1
        else:
            polarity = -1
            
        lines = cont_gen.lines(lvl)
        
        for line in lines:
            # we need to check if the contours are drawn clockwise or counterclockwise
            # https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
            x = line[:,0]
            y = line[:,1]
            
            clockwise = np.sum((x[1:] - x[:-1])*(y[1:] + y[:-1])) 
            
            if clockwise < 0:
                line = line[::-1,:]
                
            c = Contour(line.T, polarity)
            contours.append(c)
        
    return contours







def CalcContoursHat(pcb, tar_dir, seperation = 0.0003,savepath = 0, plot = False):
    x = np.linspace(-1,1, 1000) 
    y = np.linspace(-1,1, 1000)

    X, Y = np.meshgrid(x, y)
    phi = np.vectorize(pcb.curl_potential)
    Phi = phi(X,Y)     
    PhiGrad = np.array(np.gradient(Phi))
    dx = x[1]-x[0]
    max_slope = np.max(np.linalg.norm(PhiGrad, axis = 0))
    num_levels = int((np.max(Phi)-np.min(Phi))/(max_slope/dx*seperation)) #(np.max(Sol)-np.min(Sol))/min_spacing*SolGrad
    # num_levels = 10
    print((np.max(Phi)-np.min(Phi)),(max_slope/dx*seperation) )
    print("slope", max_slope)
    print("num_levels", num_levels)
    print("I", max_slope/dx*seperation)
    I = max_slope/dx*seperation

    contours = density_to_loops(Phi, num_levels, x, y)
    if plot:
        fig, ax = plt.subplots(1,1)
        fig.suptitle("Grid size {}, d = {} mm".format(pcb.M,seperation*1000))
        for c in contours:
            c.plot_contour(fig, ax)
        ax.set_aspect('equal')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if savepath:
            dist_string = "0,"+str(seperation)[2:]
            plt.savefig(savepath+"Contours_hat_{}_{}m".format(tar_dir,dist_string),dpi = 400)
    return contours, I

def CalcUniformityContours(contours, resolution,dir, I, V = [-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]):
    x = np.linspace(V[0], V[1], resolution)  # Discretize the target volume into a grid
    y = np.linspace(V[2], V[3], resolution)
    z = np.linspace(V[4], V[5], resolution)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    X = X.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
    Y = Y.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
    Z = Z.reshape((resolution ** 3), order="C")  # Order the volume lexicographically

    Bx = np.zeros_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)
    Ind = np.arange(len(X))
    for i in tqdm(Ind):
        for c in contours:
            B = I*c.calc_mag_field(np.array([X[i], Y[i], Z[i]]))
            Bx[i] += B[0]
            By[i] += B[1]
            Bz[i] += B[2]

    #Parralel script
    #B = np.zeros((len(X), 3))
    #def compute_B(i):
    #    B = np.array([0.0, 0.0, 0.0])
    #    for c in contours:
    #        B += I*c.calc_mag_field(np.array([X[i], Y[i], Z[i]]))
    #    return B
    #results = np.array(Parallel(n_jobs=-1)(delayed(compute_B)(i) for i in tqdm(range(len(X)))))
    #print(results)
    
    Bx_avg = np.mean(Bx)
    By_avg = np.mean(By) 
    Bz_avg = np.mean(Bz)
    print("Magnetic fields for lines: ",Bx_avg, By_avg, Bz_avg)
    if dir == "x":
        U = (np.sum((Bx-1)**2) + np.sum((By)**2) + np.sum((Bz)**2))/(resolution**3*1**2)
    if dir == "z":
        U = (np.sum((Bx)**2) + np.sum((By)**2) + np.sum((Bz-1)**2))/(resolution**3*1**2)
    print("U lines: ", U)
    
    if False: 
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        x2 = np.linspace(-0.5, 0.5, resolution)  # Discretize the target volume into a grid
        y2 = np.linspace(-0.5, 0.5, resolution)
        z2 = np.linspace(0.1, 1.1, resolution)
        Z2, Y2, X2 = np.meshgrid(z2, y2, x2, indexing='ij')
        X2 = X2.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
        Y2 = Y2.reshape((resolution ** 3), order="C")  # Order the volume lexicographically
        Z2 = Z2.reshape((resolution ** 3), order="C")  # Order the volume lexicographically

        B_max = max(np.linalg.norm(np.array([Bx, By, Bz]), axis = 1))
        length = 10/resolution


        ax.quiver(X2, Y2, Z2, length*Bx/B_max, length*By/B_max, length*Bz/B_max, normalize=False, color = "k")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        #ax.view_init(elev=48, azim=-65)
        ax.view_init(elev=0, azim=-90)

        plt.show()

    return Bx, By, Bz, U
