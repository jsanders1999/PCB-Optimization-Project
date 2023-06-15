
import numpy as np
import scipy as sp
import scipy.integrate as si
import scipy.linalg as li
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors


from CubeClass import Cube
from DipoleFields import *
from LexgraphicTools import *
from PCBClass import PCB_u
from OptimizeClass import optimize_k

save_path = 0
save_path = "C:/Users/aglas/OneDrive/Bureaublad/Documenten/Technische Wiskunde Master/Advanced Modelling/Project/Images/"

res = 25
cube = Cube(res)



Ms = []
Vs_x = []
Bs_x = []
Vs_z = []
Bs_z = []
Ps_x = []
Ps_z = []
count = 0
fig, axs = plt.subplots(nrows= 3,ncols = 4, layout="constrained",figsize = (16,16))
M= 4
for row in range(3):
    for col in range(4):
        pcb = PCB_u(M, None, cube, 0)

        pcb.assemble_S()
        Q = pcb.SS_x + pcb.SS_y + pcb.SS_z
        J = np.ones((pcb.cube.resolution**3,pcb.cube.resolution**3))
        R_x = pcb.S_x.T@J@pcb.S_x
        R_z = pcb.S_z.T@J@pcb.S_z

        u_x = li.solve(Q,pcb.S_x.T@np.ones(res**3))
        u_z = li.solve(Q,pcb.S_z.T@np.ones(res**3))


        Vs_x.append((res**3*u_x.T@Q@u_x)/(u_x.T@R_x@u_x) - 1)
        Bs_x.append(np.mean([1/res**3*np.sum(pcb.S_x@u_x),1/res**3*np.sum(pcb.S_y@u_x),1/res**3*np.sum(pcb.S_z@u_x)]))

        Vs_z.append((res**3*u_z.T@Q@u_z)/(u_z.T@R_z@u_z) - 1)
        Bs_z.append(np.mean([1/res**3*np.sum(pcb.S_z@u_z),1/res**3*np.sum(pcb.S_y@u_z),1/res**3*np.sum(pcb.S_z@u_z)]))
        
        Ps_x.append(u_x@u_x)
        Ps_z.append(u_z@u_z)

        Ms.append(M)
        


        # x = np.linspace(*pcb.x_bnd, 10*pcb.M + 2)[1:-1]
        # y = np.linspace(*pcb.y_bnd, 10*pcb.M + 2)[1:-1]
        # pcb.u_cart = np.reshape(u_z, (M,M), order = "C")

        # pcb.coeff_to_current(5)

        # X, Y = np.meshgrid(x, y)
        # phi = np.vectorize(pcb.curl_potential)
        # Phi = phi(X,Y)

        # pcb.calc_mag_field()
        # divnorm=colors.TwoSlopeNorm(vmin=min(np.min(Phi),-np.max(Phi)), vcenter=0., vmax=max(np.max(Phi),-np.min(Phi)))
        # ax = axs[row,col]
        # pc = ax.pcolormesh(X,Y, Phi,norm=divnorm)
        # fig.colorbar(pc,ax = ax,fraction=0.046, pad=0.04)
        # title_string = str(M) + " hat functions"
        # ax.set_title(title_string)
        # ax.set_aspect("equal")
        # M +=1

if save_path:
    plt.savefig(save_path+"Different_M_Grid_Plot_z_3x3",dpi = 400)
plt.show()

# xticks = range(Ms[0],Ms[-1],2)
# fig, ax = plt.subplots(nrows = 1,ncols = 3,figsize = (8,5),layout="constrained")
# # fig.suptitle("Non-Uniformity and Magnetic field of hat-functions")
# ax[0].plot(Ms,Vs_x, linestyle = '--', marker = '.', label = "x-direction")
# ax[0].plot(Ms,Vs_z, linestyle = '--', marker = '.', label = "z-direction")
# ax[0].set_ylabel("Non-Uniformity")
# ax[0].set_yscale('log')
# ax[0].set_xlabel("Number of hat-functions")
# ax[0].set_xticks(xticks)
# ax[0].legend()
# ax[1].plot(Ms,Bs_x,linestyle = '--', marker = '.', label = "x-direction")
# ax[1].plot(Ms,Bs_z, linestyle = '--', marker = '.', label = "z-direction")
# ax[1].set_xlabel("Number of hat-functions")
# ax[1].set_xticks(xticks)
# ax[1].set_ylabel("Magnetic Field")
# ax[1].set_yscale('log')
# ax[1].legend()
# ax[2].plot(Ms,Ps_x,linestyle = '--', marker = '.', label = "x-direction")
# ax[2].plot(Ms,Ps_z, linestyle = '--', marker = '.', label = "z-direction")
# ax[2].set_xlabel("Number of hat-functions")
# ax[2].set_xticks(xticks)
# ax[2].set_ylabel("Power")
# ax[2].set_yscale('log')
# ax[2].legend()
# if save_path:
#     plt.savefig(save_path+"Non_uni_Mag_field_Power",dpi = 400)
# plt.show()
