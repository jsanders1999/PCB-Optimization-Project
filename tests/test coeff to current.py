# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:19:08 2023

@author: jeroe
"""


import numpy as np
import scipy as sp
import scipy.integrate as si
import scipy.linalg as li
import matplotlib.pyplot as plt

from CubeClass import Cube
from PCBClass import PCB_u
from PlottingFunctions import plot_field_arrow_3d
from LexgraphicTools import reshape_array_lex_cart_sides
from OptimizeClass import optimize_k
from Contour import Contour, PCB_c, density_to_loops


plt.close('all')

N = 8
c = Cube(N)

M = 10
orientation = 0
pcb = PCB_u(M, None, c, orientation)

P   = 4         #power [W]
rho = 1         #resistivity [?]
Dz  = 0.0001    #trace thickness [m]

uNorm = (P*M*M)/(4*rho*Dz)

u_start = (uNorm/M**2)*np.ones(M**2)

u_start = (uNorm/M**2)*u_start/np.linalg.norm(u_start)


pcb.assemble_S()
Q = pcb.SS_x + pcb.SS_y + pcb.SS_z
J = np.ones((pcb.cube.resolution**3,pcb.cube.resolution**3))
R = pcb.S_z.T@J@pcb.S_z

opti_instance = optimize_k(pcb, verbose = True)
#u , Vu = opti_instance.gradient_descent_normed(u_start, 50000, 3e10, uNorm)
#u , Vu = opti_instance.line_search_line_min(u_start, 1500, uNorm/M**2)
u , Vu = opti_instance.scipy_minimum(u_start, uNorm/M**2) #Only works up to M=10

pcb.u_cart = np.reshape(u, (M,M), order = "C")
# pcb.u_cart = np.ones(pcb.u_cart.shape)

pcb.coeff_to_current(4)

fig, ax = plt.subplots(1,1)
pc2 = ax.pcolormesh(pcb.X_curr, pcb.Y_curr, pcb.potential)
plt.colorbar(pc2)
ax.set_title('potential')

# check if the grid points correctly overlap
# ax.plot(pcb.X, pcb.Y, 'b.', linestyle="None")
# ax.plot(pcb.X_curr, pcb.Y_curr, 'r.', linestyle="None")

# fig, ax = plt.subplots(1,1)
# pc2 = ax.pcolormesh(pcb.X_curr, pcb.Y_curr, pcb.current_x)
# plt.colorbar(pc2)
# ax.set_title('current x')

# fig, ax = plt.subplots(1,1)
# pc2 = ax.pcolormesh(pcb.X_curr, pcb.Y_curr, pcb.current_y)
# plt.colorbar(pc2)
# ax.set_title('current y')

num_levels = 10
pcbc = density_to_loops(pcb.potential,num_levels, c, pcb.X_curr, pcb.Y_curr)

figsize = None
fig, ax = plt.subplots(1,1, figsize = figsize)
pcbc.plot_contours(fig, ax)

pcbc.calc_mag_field()

plot_field_arrow_3d(pcbc.field, c)