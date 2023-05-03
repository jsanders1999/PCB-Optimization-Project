
import numpy as np
import scipy as sp
import scipy.integrate as si
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import json


from CubeClass import Cube
from DipoleFields import *
from LexgraphicTools import *
from PCBClass import PCB_u
from OptimizeClass import optimize_k

N = 10
M = 10

cube = Cube(N)
pcb = PCB_u(M, None, cube)

#ax = pcb.plot_field_arrow_3d()
 
P   = 4         #power [W]
rho = 1         #resistivity [?]
Dz  = 0.0001    #trace thickness [m]

uNorm = (P*M*M)/(4*rho*Dz)

u_start = (uNorm/M**2)*np.ones(M**2)
#u_start = (uNorm/M**2)*np.random.rand(M**2)

#u_start[int((M-1)**2/2+(M-1)/2)] = 400
#u_start[int((M-1)**2/2+(M-1)/2)+1] = -200

#for i in range(M):
#    for j in range(M):
#        if i<M/4 or i>3*M/4 or j<M/4 or j>3*M/4:
#            u_start[(M-1)*i+j] = 1
#u_start = (uNorm/M**2)*np.random.normal(size = M**2)
u_start = (uNorm/M**2)*u_start/np.linalg.norm(u_start)
print(np.linalg.norm(u_start), uNorm/M**2)

opti_instance = optimize_k(pcb)
u , Vu = opti_instance.gradient_descent_normed(u_start, 5000, 3e10, uNorm)
u , Vu = opti_instance.line_search_line_min(u_start, 50, uNorm/M**2)
u , Vu = opti_instance.line_search_sphere(u_start, 50, uNorm/M**2)


pcb.u_cart = np.reshape(u, (M,M), order = "C")

#pcb.plot_curl_potential()
#pcb.plot_curl_potential( contour_lvl = 10)



# print('\n',Vu)
# #print(u)
# #print(reshape_array_lex_cart(us, M)[:,:,0])
# plt.imshow(pcb.u)
# plt.colorbar()
# plt.title("u vector")
# plt.xlabel("x [m]")
# plt.ylabel("y [m]")
# plt.show()


fig1, ax1 = plt.subplots(1,1)

pc1 = ax1.contour(pcb.X, pcb.Y, pcb.u_cart, levels = 10)
plt.colorbar(pc1)
ax1.set_title("contours of u vector")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
ax1.set_aspect("equal")
plt.show()

fig2, ax2 = plt.subplots(1,1)

pc2 = ax2.pcolormesh(pcb.X, pcb.Y, pcb.u_cart)
plt.colorbar(pc2)
ax2.set_title("u vector")
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.set_aspect("equal")
plt.show()

#pcb.system_analysis()