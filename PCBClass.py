import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import json

from CubeClass import Cube
from DipoleFields import *
from LexgraphicTools import *
from hatfunctions import *


class PCB_u:
    def __init__(self, M, u_cart, cube):
        """
        Initilizes the PCB object

        Parameter
        ---------
        M : int
            Number of grid points in each direction on the PCB

        u_cart: None or np.array[M,M]
            Array with the values of u in cartesian ordering, if it is None u_cart will be made with 1 at each position

        cube: Cube
            Cube above the PCB

        Returns
        -------
        None
        """
        self.cube = cube
        self.M = M
        self.x_bnd = [-1,1]
        self.y_bnd = [-1,1]
        self.h = 2/(M +1)
        self.make_grid()

        if u_cart:
            self.u_cart = u_cart
        else:
            self.u_cart = np.ones(self.X.shape)

        self.assemble_S()
        #self.assemble_f(2)
        #self.calc_mag_field()
        

    ############################
    ### Construction Methods ###
    ############################

    def make_grid(self):
        """
        Makes the grid on the PCB
        """
        # make an array for x and y, +2 adds the end points for the creation and then the endpoints are sliced out
        self.x = np.linspace(*self.x_bnd, self.M + 2)[1:-1]
        self.y = np.linspace(*self.y_bnd, self.M + 2)[1:-1]

        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        
    def calc_mag_field_point(self, x, y, z):
        """
        Calculates the field of a dipole at the point (x,y,z)

        Parameter
        ---------
        x : float
            x-coordinate of the target point

        y : float
            y-coordinate of the target point

        z : float
            z-coordinate of the target point

        Returns
        -------
        B_x, B_y, B_z: float, float, float
            The field in the x, y and z direction
        """
        x_p = (x - self.X)
        y_p = (y - self.Y)

        # contribution of each individual dipole
        B_xp = self.h**2*B_dipole_x(x_p, y_p, z)
        B_yp = self.h**2*B_dipole_y(x_p, y_p, z)
        B_zp = self.h**2*B_dipole_z(x_p, y_p, z)

        # multiply with the weigth of each dipole
        B_x = self.u_cart*B_xp
        B_y = self.u_cart*B_yp
        B_z = self.u_cart*B_zp

        # final field
        B_x_final = np.sum(B_x)
        B_y_final = np.sum(B_y)
        B_z_final = np.sum(B_z)
        
        return B_x_final, B_y_final, B_z_final

    def assemble_S(self):
        """
        Assembles the system matrix for each direction
        """
        # length of the T vector
        NNN = self.cube.resolution**3
        # length of the u vector
        MM = self.M**2

        # Initialize
        self.S_x = np.zeros((NNN, MM))
        self.S_y = np.zeros((NNN, MM))
        self.S_z = np.zeros((NNN, MM))

        for n in tqdm(range(NNN), desc='Assembling S'):
            # Get (x,y,z) of the current point
            i,j,k = lex_to_cart_3D(n,self.cube.resolution)
            x = self.cube.X[i,j,k]
            y = self.cube.Y[i,j,k]
            z = self.cube.Z[i,j,k]

            # shift to origin 
            x_p = (x - self.X)
            y_p = (y - self.Y)

            # contribution of each individual dipole
            B_x = self.h**2*B_dipole_x(x_p, y_p, z)
            B_y = self.h**2*B_dipole_y(x_p, y_p, z)
            B_z = self.h**2*B_dipole_z(x_p, y_p, z)

            # set row of each system matrix as the flattend field array
            self.S_x[n,:] = B_x.flatten()
            self.S_y[n,:] = B_y.flatten()
            self.S_z[n,:] = B_z.flatten()

        # make square/symmetric matrix
        self.SS_x = self.S_x.T@self.S_x
        self.SS_y = self.S_y.T@self.S_y
        self.SS_z = self.S_z.T@self.S_z
        J = np.ones((NNN, NNN))
        self.Q = self.SS_x + self.SS_y + self.SS_z
        self.R = self.S_x.T@J@self.S_x + self.S_y.T@J@self.S_y + self.S_z.T@J@self.S_z 


    def calc_mag_field(self):
        """
        Calculates the field in the cube
        """
        # convert u to lex ordering
        u = self.u_cart.flatten()

        # calculate field
        B_x_flat = self.S_x@u
        B_y_flat = self.S_y@u
        B_z_flat = self.S_z@u

        # reshape arrays to cart
        N = self.cube.resolution
        B_x = reshape_array_lex_cart(B_x_flat, N)
        B_y = reshape_array_lex_cart(B_y_flat, N)
        B_z = reshape_array_lex_cart(B_z_flat, N)

        # initialize field
        self.field = np.zeros((*self.cube.X.shape,3))

        # set each direction of the field
        self.field[:,:,:,0] = B_x
        self.field[:,:,:,1] = B_y
        self.field[:,:,:,2] = B_z


    def assemble_f(self, direction):
        """
        Assembles the f vector (RHS)

        Parameter
        ---------
        direction: int in [0,1,2]
            direction of the field 0:x, 1:y, 2:z
        """
        dirx = 0
        diry = 0
        dirz = 0

        if direction == 0:
            dirx = 1
        elif direction == 1:
            diry = 1
        else:
            dirz = 1
        
        # only the specified direction of f is 1 the other directions are zero
        self.f_x = np.zeros(self.cube.resolution**3) + dirx
        self.f_y = np.zeros(self.cube.resolution**3) + diry
        self.f_z = np.zeros(self.cube.resolution**3) + dirz

        self.Sxf = self.S_x.T@self.f_x
        self.Syf = self.S_y.T@self.f_y
        self.Szf = self.S_z.T@self.f_z


    # def assemble_system(self):
    #     self.SxT = self.Sx.T@self.f


    ############################
    ###   Analysis Methods   ###
    ############################

    def system_analysis(self):
        """
        Calculate the min, max eigenvalues and the condition number of each squared system matrix
        """
        lamb, v = np.linalg.eig(self.SS_x)
        l_max_x = np.max(np.abs(lamb))
        l_min_x = np.min(np.abs(lamb))

        kappa_x = l_max_x/l_min_x

        lamb, v = np.linalg.eig(self.SS_y)
        l_max_y = np.max(np.abs(lamb))
        l_min_y = np.min(np.abs(lamb))

        kappa_y = l_max_y/l_min_y

        lamb, v = np.linalg.eig(self.SS_z)
        l_max_z = np.max(np.abs(lamb))
        l_min_z = np.min(np.abs(lamb))

        kappa_z = l_max_z/l_min_z

        self.condition = {
            "x": {
                "lambda_min": l_min_x,
                "lambda_max": l_max_x,
                "kappa": kappa_x
            },
            "y": {
                "lambda_min": l_min_y,
                "lambda_max": l_max_y,
                "kappa": kappa_y
            },
            "z": {
                "lambda_min": l_min_z,
                "lambda_max": l_max_z,
                "kappa": kappa_z
            },
        }

    ############################
    ###  Plotting Functions  ###
    ############################

    def plot_field(self, z, orientation, fig = None, ax = None):
        """"
        input: z = float between 0,1
               orientation = [0,1,2] corresponding with the x,y,z direction
        """
        self.calc_mag_field()
        label = ['x', 'y', 'z']

        if (not fig) and (not ax):
            fig, ax = plt.subplots(1,1)

        z_index = int(np.floor(z*self.cube.resolution))

        ax.set_title(f'field in the {label[orientation]}-direction')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        pc = ax.pcolormesh(self.cube.x_arr, self.cube.y_arr, self.field[:,:,z_index,orientation])
        fig.colorbar(pc)
        return fig, ax

    
    def plot_field_arrow(self, frac, orientation, fig = None, ax = None):
        """"
        input: z = float between 0,1
               orientation = [0,1,2] corresponding with the x,y,z direction
        """
        self.calc_mag_field()
        label = ['xy', 'yz', 'xz']

        if (not fig) and (not ax):
            fig, ax = plt.subplots(1,1)

        frac_index = int(np.floor(frac*self.cube.resolution))

        # ax.set_title(f'field in the {label[orientation]}-direction')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        U_x = 0.25*self.field[:,:,:,0]/np.max(self.field)
        U_y = 0.25*self.field[:,:,:,1]/np.max(self.field)
        U_z = 0.25*self.field[:,:,:,2]/np.max(self.field)
        
        if orientation == 0:
            ax.quiver(self.cube.X[:,:,frac_index], self.cube.Y[:,:,frac_index], U_x[:,:,frac_index], U_y[:,:,frac_index])
        elif orientation == 1:
            ax.quiver(self.cube.Y[:,frac_index,:], self.cube.Z[:,frac_index,:], U_y[:,frac_index,:], U_z[:,frac_index,:])
        else:
            ax.quiver(self.cube.X[frac_index,:,:], self.cube.Z[frac_index,:,:], U_x[frac_index,:,:], U_z[frac_index,:,:])
        return fig, ax


    def plot_field_arrow_3d(self, S = False, normalize = True, fig = None, ax = None):
        """"
        input:
        """
        self.calc_mag_field()
        if (not fig) and (not ax):
            ax = plt.figure().add_subplot(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        U_x = 0.25*self.field[:,:,:,0]/np.max(self.field)
        U_y = 0.25*self.field[:,:,:,1]/np.max(self.field)
        U_z = 0.25*self.field[:,:,:,2]/np.max(self.field)
        ax.quiver(self.cube.X, self.cube.Y, self.cube.Z, U_x, U_y, U_z)
        return ax
    
    def K_ij_x(self, x_s, y_s, i, j):
        return K0_x((x_s-self.x[i])/self.h, (y_s-self.y[j])/self.h )

    def K_ij_y(self, x_s, y_s, i, j):
        return K0_y((x_s-self.x[i])/self.h, (y_s-self.y[j])/self.h )

    def K_x(self, x_s, y_s):
        I, J = np.meshgrid(range(self.M), range(self.M))
        return np.sum(self.u_cart[I, J]    *  self.K_ij_x(x_s, y_s, I, J))
    
    def K_y(self, x_s, y_s):
        I, J = np.meshgrid(range(self.M), range(self.M))
        return np.sum(self.u_cart[I, J]    *  self.K_ij_y(x_s, y_s, I, J))

    def K(self, x_s, y_s):
        return np.array(self.K_x(x_s, y_s), self.K_y(x_s, y_s))
    
    def curl_potential_ij(self, x_s, y_s, i, j):
        return self.h*curl_potential_0((x_s-self.x[i])/self.h, (y_s-self.y[j])/self.h )

    def curl_potential(self, x_s, y_s):
        I, J = np.meshgrid(range(self.M), range(self.M))
        return np.sum(self.u_cart[I, J]    *  self.curl_potential_ij(x_s, y_s, I, J))

    def plot_K(self, fig = None, ax = None):
        if (not fig) and (not ax):
            fig1, ax1 = plt.subplots(1,1)
            fig2, ax2 = plt.subplots(1,1)
            fig3, ax3 = plt.subplots(1,1)
        #Must be of a higher resoltion than self.x and not on the same points
        x = np.linspace(*self.x_bnd, 10*self.M + 2)[1:-1]
        y = np.linspace(*self.y_bnd, 10*self.M + 2)[1:-1]

        X, Y = np.meshgrid(x, y)
        K_x = np.vectorize(self.K_x)
        K_y = np.vectorize(self.K_y)
        K_X = K_x(X, Y) 
        K_Y = K_y(X, Y) 
        ax1.quiver(X, Y, K_X, K_Y)
        ax1.set_aspect('equal')
        ax2.imshow(K_X)
        ax3.imshow(K_Y)
        return fig, ax
    
    def plot_curl_potential(self, fig = None, ax = None, contour_lvl = False):
        if (not fig) and (not ax):
            fig, ax = plt.subplots(1,1)
        #Must be of a higher resoltion than self.x and not on the same points
        x = np.linspace(*self.x_bnd, 10*self.M + 2)[1:-1]
        y = np.linspace(*self.y_bnd, 10*self.M + 2)[1:-1]

        X, Y = np.meshgrid(x, y)
        phi = np.vectorize(self.curl_potential)
        Phi = phi(X,Y)
        if contour_lvl:
            pc = ax.contour(X, Y, Phi, levels = contour_lvl)
            fig.colorbar(pc)
        else:
            pc = ax.pcolormesh(X,Y, Phi)
            fig.colorbar(pc)
        return fig, ax