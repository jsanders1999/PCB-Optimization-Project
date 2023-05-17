# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:10:43 2023

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

opti_instance = optimize_k(pcb, verbose = False)
#u , Vu = opti_instance.gradient_descent_normed(u_start, 50000, 3e10, uNorm)
#u , Vu = opti_instance.line_search_line_min(u_start, 1500, uNorm/M**2)
u , Vu = opti_instance.scipy_minimum(u_start, uNorm/M**2) #Only works up to M=10

pcb.u_cart = np.reshape(u, (M,M), order = "C")

pcb.calc_mag_field()

plot_field_arrow_3d(pcb.field, c)

# not the exact solution but the implementation only needs to a
potential = pcb.u_cart


fig1, ax1 = plt.subplots(1,1)

pc1 = ax1.contour(pcb.X, pcb.Y, pcb.u_cart, levels = 10)
plt.colorbar(pc1)
ax1.set_title("contours of u vector")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
ax1.set_aspect("equal")
plt.show()

num_levels = 12

pcbc = density_to_loops(potential,num_levels, c, pcb)

figsize = (9,9)
fig, ax = plt.subplots(1,1, figsize = figsize)
pcbc.plot_contours(fig, ax)

pcbc.calc_mag_field()

plot_field_arrow_3d(pcbc.field, c)

plot_field_arrow_3d(pcb.field - pcbc.field, c)

