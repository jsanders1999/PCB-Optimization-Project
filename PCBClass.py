import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import json
import scipy.sparse as sp


from CubeClass import Cube
from DipoleFields import *
from LexgraphicTools import *
from hatfunctions import *

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class PCB_u:
    def __init__(self, M, u_cart, cube, orientation, x_bnd = [-1,1], y_bnd = [-1,1]):
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
            
        orientation: int
            0 or 1 denoting if the uniform field needs to be in the z or x direction
            y direction is not considered since it is just a rotation of the x direction
        
        Returns
        -------
        None
        """
        self.cube = cube
        self.M = M
        self.x_bnd = x_bnd
        self.y_bnd = y_bnd
        self.h = 2/(M +1)
        self.make_grid()
        

        if u_cart:
            self.u_cart = u_cart
        else:
            self.u_cart = np.ones(self.X.shape)

        # self.assemble_S()
        # self.assemble_f(2)
        # self.calc_mag_field()
        
        self.u_cart_symm = np.ones(self.X.shape)
        self.orientation = orientation
        # sign = 1 if orientation = 0 and sign = -1 if orientation = 1
        self.symm_sign = 1 -2*self.orientation 
        

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
        
        # first index is row: y, second index is col: x
        self.Y, self.X = np.meshgrid(self.y, self.x)
        
        self.x_symm = np.linspace(*self.x_bnd, 2*self.M + 2)[1:self.M + 1]
        self.y_symm = np.linspace(*self.y_bnd, 2*self.M + 2)[1:self.M + 1]
        
        # first index is row: y, second index is col: x
        self.Y_symm, self.X_symm = np.meshgrid(self.y_symm, self.x_symm)
        
        
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
        B_xp = self.h**2*B_dipole_x(x_p, y_p, z,1/self.M)
        B_yp = self.h**2*B_dipole_y(x_p, y_p, z,1/self.M)
        B_zp = self.h**2*B_dipole_z(x_p, y_p, z,1/self.M)

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
            B_x = self.h**2*B_dipole_x(x_p, y_p, z,1/self.M)
            B_y = self.h**2*B_dipole_y(x_p, y_p, z,1/self.M)
            B_z = self.h**2*B_dipole_z(x_p, y_p, z,1/self.M)

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


    def expand_u_symm(self):
        old_shape = np.array(self.u_cart_symm.shape)
        new_shape = old_shape*2 
        self.u_cart_exp = np.zeros(new_shape)
        
        self.u_cart_exp[:old_shape[0], :old_shape[1]] = self.u_cart_symm
        
        # for some reason [:-1:-1,:-1] does not work thus this work around is needed
        self.u_cart_exp[old_shape[0]:, :old_shape[1]] += (self.u_cart_symm[::-1,:])#[::-1,:]
        
        self.u_cart_exp[:,old_shape[1]:] = self.symm_sign*(self.u_cart_exp[:,:old_shape[1]])[:,::-1]


    def assemble_S_symm(self):
        """
        Assembles the system matrix for each direction
        """
        # length of the T vector
        NNN = self.cube.resolution**2*6
        # length of the u vector
        MM = self.M**2

        # Initialize
        self.S_symm_x = np.zeros((NNN, MM))
        self.S_symm_y = np.zeros((NNN, MM))
        self.S_symm_z = np.zeros((NNN, MM))

        for n in tqdm(range(NNN), desc='Assembling S symmetric'):
            # Get (x,y,z) of the current point
            i,j,k = lex_to_cart_3D(n,self.cube.resolution)
            
            # k is the slow moving variable which will denote the respective side
            x = self.cube.sides_X[k,i,j]
            y = self.cube.sides_Y[k,i,j]
            z = self.cube.sides_Z[k,i,j]

            # shift to origin 
            x_p = (x - self.X_symm)
            y_p = (y - self.Y_symm)

            # contribution of each individual dipole
            B_x = self.h**2*B_dipole_x(x_p, y_p, z,1/self.M)
            B_y = self.h**2*B_dipole_y(x_p, y_p, z,1/self.M)
            B_z = self.h**2*B_dipole_z(x_p, y_p, z,1/self.M)

            # set row of each system matrix as the flattend field array
            self.S_symm_x[n,:] = B_x.flatten()
            self.S_symm_y[n,:] = B_y.flatten()
            self.S_symm_z[n,:] = B_z.flatten()

        # make square/symmetric matrix
        self.SS_symm_x = self.S_symm_x.T@self.S_symm_x
        self.SS_symm_y = self.S_symm_y.T@self.S_symm_y
        self.SS_symm_z = self.S_symm_z.T@self.S_symm_z
        J = np.ones((NNN, NNN))
        self.Q_symm = self.SS_symm_x + self.SS_symm_y + self.SS_symm_z
        self.R_symm = self.S_symm_x.T@J@self.S_symm_x + self.S_symm_y.T@J@self.S_symm_y + self.S_symm_z.T@J@self.S_symm_z 

    def assemble_A(self, thick = 70e-6,resis = 1.7241e-8):
        '''
        Constructs Power matrix
        '''
        h = (self.x_bnd[1]-self.x_bnd[0])/self.M
        diag = [8/3*h**2*np.ones(self.M)]
        L = sp.diags(diag,[0],shape=(self.M,self.M))
        I = sp.eye(self.M)
        diags2 = [-1/3*h**2*np.ones(self.M-1),-1/3*h**2*np.ones(self.M-1)]
        L2 = sp.diags(diags2,[-1,1],shape=(self.M,self.M))
        i = np.ones((self.M-1))
        I2 = sp.diags([i,i],[-1,1],shape=(self.M,self.M))
        A = sp.kron(I,L)+sp.kron(L,I)+ sp.kron(I2,L2)
        self.A = (resis/thick)*(A.A)


    def calc_mag_field_symm(self):
        """
        Calculates the field in the cube
        """
        # convert u to lex ordering
        u = self.u_cart_symm.flatten()

        # calculate field
        B_x_flat = self.S_symm_x@u
        B_y_flat = self.S_symm_y@u
        B_z_flat = self.S_symm_z@u

        # reshape arrays to cart
        N = self.cube.resolution
        B_x = reshape_array_lex_cart_sides(B_x_flat, N)
        B_y = reshape_array_lex_cart_sides(B_y_flat, N)
        B_z = reshape_array_lex_cart_sides(B_z_flat, N)

        # initialize field
        self.field_symm = np.zeros((*self.cube.sides_X.shape,3))

        # set each direction of the field
        self.field_symm[:,:,:,0] = B_x
        self.field_symm[:,:,:,1] = B_y
        self.field_symm[:,:,:,2] = B_z
        
        # the calculated field is only from the lower left corner
        # use symmetries to add the fields, first from the upper left corner
        # the field at the x sides are summed with itself reversed in the y direction
        # and the field in the y direction needs to be flipped
        self.field_symm[0:2,:,:,0] += np.flip(self.field_symm[0:2,:,:,0],1)
        self.field_symm[0:2,:,:,1] += -np.flip(self.field_symm[0:2,:,:,1],1)
        self.field_symm[0:2,:,:,2] += np.flip(self.field_symm[0:2,:,:,2],1)
        
        # the field at the y sides are summed with the field of the opposite y side
        self.field_symm[2:4,:,:,0] += np.flip(self.field_symm[2:4,:,:,0],0)
        self.field_symm[2:4,:,:,1] += -np.flip(self.field_symm[2:4,:,:,1],0)
        self.field_symm[2:4,:,:,2] += np.flip(self.field_symm[2:4,:,:,2],0)
        
        # the field at the z sides are summed with itself reversed in the y direction
        self.field_symm[4:6,:,:,0] += np.flip(self.field_symm[4:6,:,:,0],1)
        self.field_symm[4:6,:,:,1] += -np.flip(self.field_symm[4:6,:,:,1],1)
        self.field_symm[4:6,:,:,2] += np.flip(self.field_symm[4:6,:,:,2],1)
        
        # use the symmetries to add the fields from the right half
        # for each addition the symm_sign is used
        # the field at the x sides are summed with the field of the opposite x side
        # and the field in the x direction needs to be flipped
        self.field_symm[0:2,:,:,0] += -self.symm_sign*np.flip(self.field_symm[0:2,:,:,0],0)
        self.field_symm[0:2,:,:,1:3] += self.symm_sign*np.flip(self.field_symm[0:2,:,:,1:3],0)
        
        # the field at the y sides are summed with itself reversed in the x direction
        # the second index is the x index 
        self.field_symm[2:4,:,:,0] += -self.symm_sign*self.field_symm[2:4,::-1,:,0]
        self.field_symm[2:4,:,:,1:3] += self.symm_sign*self.field_symm[2:4,::-1,:,1:3]
        
        # the field at the z sides are summed with itself reversed in the x direction
        self.field_symm[4:6,:,:,0] += -self.symm_sign*self.field_symm[4:6,:,::-1,0]
        self.field_symm[4:6,:,:,1:3] += self.symm_sign*self.field_symm[4:6,:,::-1,1:3]
        
        
    def coeff_to_current(self, pnts_in_el):
        # make a finer grid to plot, the number of points is the number of elements (M+1) 
        # times the amount of pnts in each element, 1 is added to include the end point
        self.x_curr = np.linspace(*self.x_bnd, pnts_in_el*(self.M+1)+1)
        self.y_curr = np.linspace(*self.y_bnd, pnts_in_el*(self.M+1)+1)
        self.Y_curr, self.X_curr = np.meshgrid(self.y_curr,self.x_curr)
        
        # initialize arrays
        self.current_x = np.zeros(self.X_curr.shape)
        self.current_y = np.zeros(self.X_curr.shape)
        
        self.potential = np.zeros(self.X_curr.shape)
        
        # collection of shifts to the closest grid points of the hat functions
        add = [[-1,-1], [0,-1], [-1,0], [0,0]]
        # loop over the elements
        # ind and jnd are indexes of each element
        for ind in range(self.M+1):
            for jnd in range(self.M+1):
                # transform the element wise index to starting point of the global index of all the points
                i = ind*pnts_in_el
                j = jnd*pnts_in_el
                
                # loop over the grid points of interest
                for a in add:
                    # if the element is a boundary element then the points which need to be taken
                    # do not exist so it fails and continues to the next point
                    try:
                        # make a matrix of all the points in the element and shift it to the 
                        # reference element
                        x = self.X_curr[i:i+pnts_in_el,j:j+pnts_in_el] - self.x[ind + a[0]]
                        y = self.Y_curr[i:i+pnts_in_el,j:j+pnts_in_el] - self.y[jnd + a[1]]
                        
                        # calculate the current on the points
                        curr = current_lambda(x, y, self.h)
                        
                        # add the current times the respective coefficients to the final matrix
                        self.current_x[i:i+pnts_in_el,j:j+pnts_in_el] += self.u_cart[ind + a[0],jnd + a[1]]*curr[0]
                        self.current_y[i:i+pnts_in_el,j:j+pnts_in_el] += self.u_cart[ind + a[0],jnd + a[1]]*curr[1]
                        
                        # add the potential times the respective coefficients to the final matrix
                        self.potential[i:i+pnts_in_el,j:j+pnts_in_el] += self.u_cart[ind + a[0],jnd + a[1]]*lambda_2D(x,y,self.h)
                    except:
                        continue
        
        
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

        fig.suptitle(f'field in the {label[orientation]}-direction')

        U_x = 0.25*self.field[:,:,:,0]/np.max(self.field)
        U_y = 0.25*self.field[:,:,:,1]/np.max(self.field)
        U_z = 0.25*self.field[:,:,:,2]/np.max(self.field)
        
        if orientation == 0:
            ax.quiver(self.cube.X[:,:,frac_index], self.cube.Y[:,:,frac_index], U_x[:,:,frac_index], U_y[:,:,frac_index])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        elif orientation == 1:
            ax.quiver(self.cube.Y[:,frac_index,:], self.cube.Z[:,frac_index,:], U_y[:,frac_index,:], U_z[:,frac_index,:])
            max_arr = 0
            ax.set_xlabel('y')
            ax.set_ylabel('z')
        else:
            ax.quiver(self.cube.X[frac_index,:,:], self.cube.Z[frac_index,:,:], U_x[frac_index,:,:], U_z[frac_index,:,:])
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        return fig, ax
    
    def plot_field_arrow_all_orien(self, fracs, fig = None, ax = None,save_path = False):
        """"
        input: fracs = floats between 0,1, one for each orintation
        """
        self.calc_mag_field()
        label = ['xy', 'yz', 'xz']
        frac_index_z = int(np.floor(fracs[0]*self.cube.resolution))
        frac_index_y = int(np.floor(fracs[1]*self.cube.resolution))
        frac_index_x = int(np.floor(fracs[2]*self.cube.resolution))

        U_x = 0.25*self.field[:,:,:,0]/np.max(self.field)
        U_y = 0.25*self.field[:,:,:,1]/np.max(self.field)
        U_z = 0.25*self.field[:,:,:,2]/np.max(self.field)

        max_arr = 0
        for i in range(U_x.shape[0]):
            for j in range(U_y.shape[1]):
                arr_len_xy = np.sqrt(U_x[i,j,frac_index_z]**2+U_y[i,j,frac_index_z]**2)
                arr_len_xz = np.sqrt(U_x[frac_index_y,i,j]**2+U_z[frac_index_y,i,j]**2)
                arr_len_yz = np.sqrt(U_y[i,frac_index_x,j]**2+U_z[i,frac_index_x,j]**2)
                arr_len_total = max(arr_len_xy,arr_len_xz,arr_len_yz)
                if arr_len_total>max_arr:
                    max_arr = arr_len_total

        if (not fig) and (not ax):
            fig, (ax_z,ax_y,ax_x) = plt.subplots(1,3,figsize = (16,5))

        inchscale = (self.cube.resolution/3)*max_arr
        cmscale = 1/2.54 * (inchscale) 
        fig.suptitle("Field in all 3 directions \n"+ r"$\uparrow$: 1 cm = {0:.4f} ".format(cmscale))

        ax_z.quiver(self.cube.X[:,:,frac_index_z], self.cube.Y[:,:,frac_index_z], U_x[:,:,frac_index_z], U_y[:,:,frac_index_z],scale=inchscale, scale_units='inches')
        ax_z.set_title("(x,y)-plane, z = {0:.2f}".format(frac_index_z/self.cube.resolution+0.1))
        ax_z.set_xlabel('x [m]')
        ax_z.set_ylabel('y [m]')
        ax_z.set_box_aspect(1)

        ax_y.quiver(self.cube.X[frac_index_y,:,:], self.cube.Z[frac_index_y,:,:], U_x[frac_index_y,:,:], U_z[frac_index_y,:,:],scale=inchscale, scale_units='inches')
        ax_y.set_title("(x,z)-plane, y = {0:.2f}".format(frac_index_y/self.cube.resolution))
        ax_y.set_xlabel('x [m]')
        ax_y.set_ylabel('z [m]')
        ax_y.set_box_aspect(1)

        ax_x.quiver(self.cube.Y[:,frac_index_x,:], self.cube.Z[:,frac_index_x,:], U_y[:,frac_index_x,:], U_z[:,frac_index_x,:],scale=inchscale, scale_units='inches')
        ax_x.set_title("(y,z)-plane, x = {0:.2f}".format(frac_index_x/self.cube.resolution))
        ax_x.set_xlabel('y [m]')
        ax_x.set_ylabel('z [m]')
        ax_x.set_box_aspect(1)

        if save_path:
            plt.savefig(save_path+"Arrow_all_dir",dpi = 400)


    def plot_field_arrow_extra(self, fracs, orientations, fig = None, ax = None,save_path = False):
        """"
        input: z = float between 0,1
               orientation = [0,1,2] corresponding with the x,y,z direction
        """
        self.calc_mag_field()
        label = ['xy', 'yz', 'xz']
        frac_indices = []
        for k in range(len(fracs)):
            frac_indices.append(int(np.floor(fracs[k]*self.cube.resolution)))

        U_x = 0.25*self.field[:,:,:,0]/np.max(self.field)
        U_y = 0.25*self.field[:,:,:,1]/np.max(self.field)
        U_z = 0.25*self.field[:,:,:,2]/np.max(self.field)

        max_arr = 0
        for i in range(U_x.shape[0]):
            for j in range(U_y.shape[1]):
                for k in range(len(fracs)):
                    if orientations[k]==0:
                        arr_len = np.sqrt(U_x[i,j,frac_indices[k]]**2+U_y[i,j,frac_indices[k]]**2)
                    elif orientations[k]==1:
                        arr_len = np.sqrt(U_x[frac_indices[k],i,j]**2+U_y[frac_indices[k],i,j]**2)
                    elif orientations[k]==2:
                        arr_len = np.sqrt(U_x[i,frac_indices[k],j]**2+U_y[i,frac_indices[k],j]**2)
                    else:
                        raise Exception("Orientations should be 0,1 or 2")
                    if arr_len>max_arr:
                        max_arr = arr_len


        figs = int(np.ceil(len(fracs)/3))
        if (not fig) and (not ax):
            fig, axes = plt.subplots(nrows = figs,ncols = 3,figsize = (12,figs*3+3),layout="constrained")
        

        inchscale = (8*self.cube.resolution)*max_arr
        cmscale = 1/2.54 * (inchscale) 
        fig.suptitle("Field in all 3 directions \n"+ r"$\uparrow$: 1 cm = {0:.4f} ".format(cmscale))

        k = 0
        for row in axes:
            for col in row:
                if orientations[k] == 0:
                    col.quiver(self.cube.X[:,:,frac_indices[k]], self.cube.Y[:,:,frac_indices[k]], U_x[:,:,frac_indices[k]], U_y[:,:,frac_indices[k]],scale=inchscale, scale_units='inches')
                    col.set_title("(x,y)-plane, z = {0:.2f}".format(frac_indices[k]/self.cube.resolution+0.1))
                    col.set_xlabel('x [m]')
                    col.set_ylabel('y [m]')
                    col.set_box_aspect(1)
                elif orientations[k] == 1:
                    col.quiver(self.cube.X[frac_indices[k],:,:], self.cube.Z[frac_indices[k],:,:], U_x[frac_indices[k],:,:], U_z[frac_indices[k],:,:],scale=inchscale, scale_units='inches')
                    col.set_title("(x,z)-plane, y = {0:.2f}".format(frac_indices[k]/self.cube.resolution))
                    col.set_xlabel('x [m]')
                    col.set_ylabel('z [m]')
                    col.set_box_aspect(1)
                elif orientations[k] == 2:
                    col.quiver(self.cube.Y[:,frac_indices[k],:], self.cube.Z[:,frac_indices[k],:], U_y[:,frac_indices[k],:], U_z[:,frac_indices[k],:],scale=inchscale, scale_units='inches')
                    col.set_title("(y,z)-plane, x = {0:.2f}".format(frac_indices[k]/self.cube.resolution))
                    col.set_xlabel('y [m]')
                    col.set_ylabel('z [m]')
                    col.set_box_aspect(1)
                k+=1
        if save_path:
            plt.savefig(save_path+"Arrow_large",dpi = 400)


    def plot_field_arrow_3d(self, S = False, normalize = True, fig = None, ax = None):
        """"
        input:
        """
        self.calc_mag_field()

        U_x = 0.25*self.field[:,:,:,0]/np.max(self.field)
        U_y = 0.25*self.field[:,:,:,1]/np.max(self.field)
        U_z = 0.25*self.field[:,:,:,2]/np.max(self.field)


        max_arr = 0
        for i in range(U_x.shape[0]):
            for j in range(U_x.shape[1]):
                for k in range(U_x.shape[2]):
                    arr_len = np.sqrt(U_x[i,j,k]**2+U_y[i,j,k]**2+U_z[i,j,k]**2)
                    if arr_len>max_arr:
                        max_arr = arr_len

        if (not fig) and (not ax):
            ax = plt.figure().add_subplot(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.quiver(self.cube.X, self.cube.Y, self.cube.Z, U_x, U_y, U_z,arrow_length_ratio = 1/self.cube.resolution*max_arr)
        return ax
    
    def PlotSol3D_hatfunctions(self,u,save_path= 0,view = 1):
        # Create 3D plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        x = np.linspace(-1,1, 1000) 
        y = np.linspace(-1,1, 1000)
        X, Y = np.meshgrid(x, y)
        Sol = np.zeros(X.shape)

        X, Y = np.meshgrid(x, y)
        phi = np.vectorize(self.curl_potential)
        Phi = phi(X,Y)

        # Plot the flat plane on the XY plane
        ax.plot_surface(X, Y, np.zeros_like(Phi), facecolors=plt.cm.viridis((Phi -np.min(Phi))/(np.max(Phi)-np.min(Phi))), shade=False, antialiased=False, zorder=-1)

        V = [-0.5, 0.5, -0.5, 0.5, 0.1, 1.1]

        vertices = [
            [V[0], V[2], V[4]],
            [V[1], V[2], V[4]],
            [V[1], V[3], V[4]],
            [V[0], V[3], V[4]],
            [V[0], V[2], V[5]],
            [V[1], V[2], V[5]],
            [V[1], V[3], V[5]],
            [V[0], V[3], V[5]]
        ]

        edges = [
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[3]],
            [vertices[3], vertices[0]],
            [vertices[4], vertices[5]],
            [vertices[5], vertices[6]],
            [vertices[6], vertices[7]],
            [vertices[7], vertices[4]],
            [vertices[0], vertices[4]],
            [vertices[1], vertices[5]],
            [vertices[2], vertices[6]],
            [vertices[3], vertices[7]]
        ]
        ax.set_zlim(-0.5,1.5)
        ax.set_aspect("equal")

        edge_collection = Poly3DCollection(edges, linewidths=2.5, edgecolors='red')
        ax.add_collection(edge_collection)
        #ax.set_xlim(-1.25,1.25)
        #ax.set_ylim(-1.25,1.25)

        x2 = np.linspace(V[0], V[1], self.cube.resolution)  # Discretize the target volume into a grid
        y2 = np.linspace(V[2], V[3], self.cube.resolution)
        z2 = np.linspace(V[4], V[5], self.cube.resolution)
        Z2, Y2, X2 = np.meshgrid(z2, y2, x2, indexing='ij')
        X2 = X2.reshape((self.cube.resolution ** 3), order="C")  # Order the volume lexicographically
        Y2 = Y2.reshape((self.cube.resolution ** 3), order="C")  # Order the volume lexicographically
        Z2 = Z2.reshape((self.cube.resolution ** 3), order="C")  # Order the volume lexicographically

        Bx = self.S_x@u
        By = self.S_y@u
        Bz = self.S_z@u
        B_max = max(np.linalg.norm(np.array([Bx, By, Bz]), axis = 1))
        length = 10/self.cube.resolution


        ax.quiver(X2, Y2, Z2, length*Bx/B_max, length*By/B_max, length*Bz/B_max, normalize=False, color = "k")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        if save_path and view == 1:
            ax.view_init(elev=0, azim=-90)
            plt.savefig(save_path+"3D_flat",dpi = 400)
        if save_path and view == 2:
            ax.view_init(elev=48, azim=-65)
            plt.savefig(save_path+"3D_high",dpi = 400)
        return fig, ax

    
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
    
    def plot_curl_potential(self, uniformity= 0, B_tot = 0, fig = None, ax = None, contour_lvl = False,save_path = False):
        if (not fig) and (not ax):
            fig, ax = plt.subplots(1,1)
        #Must be of a higher resoltion than self.x and not on the same points
        x = np.linspace(*self.x_bnd, 10*self.M + 2)[1:-1]
        y = np.linspace(*self.y_bnd, 10*self.M + 2)[1:-1]

        X, Y = np.meshgrid(x, y)
        phi = np.vectorize(self.curl_potential)
        Phi = phi(X,Y)
        if contour_lvl:
            pc = ax.contour(X, Y, Phi, levels = contour_lvl,cmap = 'seismic')
            fig.colorbar(pc)
            # ax.set_title("Wire Paths")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal")
            if save_path:
                plt.savefig(save_path+"Contour",dpi = 400)


        else:
            self.calc_mag_field()
            pc = ax.pcolormesh(X,Y, Phi)
            fig.colorbar(pc)
            title_string = "Current Potential for N = {0}, res = {1},\n V = {2:.3e}, B = {3:.3e}".format(self.M, self.cube.resolution, uniformity,B_tot)
            ax.set_title(title_string)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal")
            if save_path:
                plt.savefig(save_path+"Curl_Potential",dpi = 400)


        return fig, ax