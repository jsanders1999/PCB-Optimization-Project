
import numpy as np
import scipy as sp
import scipy.integrate as si
import scipy.linalg as li
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import json


from CubeClass import Cube
from DipoleFields import *
from LexgraphicTools import *
from PCBClass import PCB_u
from OptimizeClass import optimize_k

res = 30
M = 20

cube = Cube(res)
pcb = PCB_u(M, None, cube, 0)

#ax = pcb.plot_field_arrow_3d()
 
P   = 4         #power [W]
rho = 1         #resistivity [?]
Dz  = 0.0001    #trace thickness [m]

uNorm = (P*M*M)/(4*rho*Dz)

pcb.assemble_S()
Q = pcb.SS_x + pcb.SS_y + pcb.SS_z
J = np.ones((pcb.cube.resolution**3,pcb.cube.resolution**3))
R = pcb.S_z.T@J@pcb.S_z


u = li.solve(Q,pcb.S_z.T@np.ones(res**3))

u = uNorm/li.norm(u)*u


V = (res**3*u.T@Q@u)/(u.T@R@u) - 1
print("Uniformity : ", V)
#Plotting



pcb.u_cart = np.reshape(u, (M,M), order = "C")

pcb.coeff_to_current(5)


## Figure 1 Imshow u
fig3, ax3 = plt.subplots(1,1)

pc2 = ax3.pcolormesh(pcb.X, pcb.Y, pcb.u_cart)
plt.colorbar(pc2)
ax3.set_title("u vector")
ax3.set_xlabel("x [m]")
ax3.set_ylabel("y [m]")
ax3.set_aspect("equal")


##Figure 2 Potential
fig2, ax2 = plt.subplots(1,1)
plt.pcolormesh(pcb.X_curr, pcb.Y_curr, pcb.potential)
plt.title("Potential")

##Figure 3 Curl Potential colormesh (visible tents)
pcb.plot_curl_potential()
## Figure 4 Current Loops 
pcb.plot_curl_potential(contour_lvl = 10)

pcb.plot_field_arrow(0.1,0)
pcb.plot_field_arrow(0.1,1)
pcb.plot_field_arrow(0.1,2)

pcb.system_analysis()
print(pcb.condition)
plt.show()