import numpy as np

class Cube:
    def __init__(self, resolution, x = [-0.5, 0.5], y = [-0.5, 0.5], z = [0.2, 1.2]):
        """
        Initialize the cube object

        Parameter
        ---------
        resolution : int
            number of gird points in each direction

        x: list, default = [-0.5, 0.5]
            list with the lower and upper value of the x-domain
        
        y: list, default = [-0.5, 0.5]
            list with the lower and upper value of the y-domain

        z: list, default = [0.1, 1]
            list with the lower and upper value of the z-domain
        
        Returns
        -------
        None
        """
        self.resolution = resolution
        self.x_bnd = x
        self.y_bnd = y
        self.z_bnd = z
        
        self.x_arr = np.linspace(*x, resolution)
        self.y_arr = np.linspace(*y, resolution)
        self.z_arr = np.linspace(*z, resolution)

        # first index is row: y, second index is col: x
        self.Y, self.X, self.Z = np.meshgrid(self.y_arr, self.x_arr, self.z_arr, indexing="ij")
    
        self.make_sides()
         
    
    def make_sides(self):
        # make the sides of the cube
        self.sides_X = np.zeros((6,self.resolution,self.resolution))
        self.sides_Y = np.zeros((6,self.resolution,self.resolution))
        self.sides_Z = np.zeros((6,self.resolution,self.resolution))
        
        # The sides are arranged as
        # pos x, neg x, pos y, neg y, pos z, neg z
        self.sides_X[0] = self.X[:,-1,:]
        self.sides_Y[0] = self.Y[:,-1,:]
        self.sides_Z[0] = self.Z[:,-1,:]
        
        self.sides_X[1] = self.X[:,0,:]
        self.sides_Y[1] = self.Y[:,0,:]
        self.sides_Z[1] = self.Z[:,0,:]
        
        self.sides_X[2] = self.X[-1,:,:]
        self.sides_Y[2] = self.Y[-1,:,:]
        self.sides_Z[2] = self.Z[-1,:,:]
        
        self.sides_X[3] = self.X[0,:,:]
        self.sides_Y[3] = self.Y[0,:,:]
        self.sides_Z[3] = self.Z[0,:,:]
        
        self.sides_X[4] = self.X[:,:,-1]
        self.sides_Y[4] = self.Y[:,:,-1]
        self.sides_Z[4] = self.Z[:,:,-1]
        
        self.sides_X[5] = self.X[:,:,0]
        self.sides_Y[5] = self.Y[:,:,0]
        self.sides_Z[5] = self.Z[:,:,0]
        
       