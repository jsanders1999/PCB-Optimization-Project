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