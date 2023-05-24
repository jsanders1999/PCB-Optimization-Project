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

res = 10
M = 3

cube = Cube(res)
pcb = PCB_u(M, None, cube, 0)

 
P   = 4         #power [W]
rho = 1         #resistivity [?]
Dz  = 0.0001    #trace thickness [m]

uNorm = (P*M*M)/(4*rho*Dz)

# u_start = (uNorm/M**2)*np.ones(M**2)
u_start = (uNorm/M**2)*np.random.rand(M**2)


u_start = (uNorm/M**2)*u_start/np.linalg.norm(u_start)
print("Norm u_start",np.linalg.norm(u_start), uNorm/M**2)

pcb.assemble_S()
Q = pcb.SS_x + pcb.SS_y + pcb.SS_z
J = np.ones((pcb.cube.resolution**3,pcb.cube.resolution**3))
R = pcb.S_z.T@J@pcb.S_z

nSpace = li.null_space(Q)
print("Null Space shape:", nSpace.shape)
    
opti_instance = optimize_k(pcb)
# u1 , Vu1 = opti_instance.gradient_descent_normed(u_start, 50000, 3e10, uNorm)
# u2 , Vu2 = opti_instance.line_search_line_min(u_start, 1500, uNorm/M**2)
u3 , Vu3, optres = opti_instance.scipy_minimum_sphere(u_start, uNorm/M**2) #Only works up to M=10
# u4 , Vu4 = opti_instance.line_search_sphere(u_start, 50, uNorm/M**2)

print(np.linalg.norm(optres.jac)) 
print(optres)



# print("Final uniformity", Vu3)
# print("Jacobian norm ", np.linalg.norm(optres.jac))
#Plotting


##Figure 1 Imshow
fig1, ax1 = plt.subplots(1,1)
ax1.imshow(u3.reshape((M,M)))

pcb.u_cart = np.reshape(u3, (M,M), order = "C")

pcb.coeff_to_current(5)


##Figure 2
fig2, ax2 = plt.subplots(1,1)
plt.pcolormesh(pcb.X_curr, pcb.Y_curr, pcb.potential)

##Figure 3 Current Potential
pcb.plot_curl_potential()
## Figure 4 Current Loops
pcb.plot_curl_potential( contour_lvl = 10)


##Figure 5
fig3, ax3 = plt.subplots(1,1)

pc2 = ax3.pcolormesh(pcb.X, pcb.Y, pcb.u_cart)
plt.colorbar(pc2)
ax3.set_title("u vector")
ax3.set_xlabel("x [m]")
ax3.set_ylabel("y [m]")
ax3.set_aspect("equal")
plt.show()

pcb.system_analysis()