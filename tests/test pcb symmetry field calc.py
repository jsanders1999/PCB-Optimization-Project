# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:58:00 2023

@author: jeroe
"""


import numpy as np
import matplotlib.pyplot as plt

from CubeClass import Cube
from PCBClass import PCB_u
from PlottingFunctions import plot_field_arrow_3d
from LexgraphicTools import reshape_array_lex_cart_sides


plt.close('all')

N = 8
c = Cube(N)

M = 10
orientation = 0
p = PCB_u(M, None, c, orientation)

# increasing = np.linspace(1,0.1,M)
# test = np.outer(increasing, increasing.T)

# p.u_cart_symm = test

# p.expand_u_symm()

# res = p.u_cart_exp 

p.assemble_S_symm()

p.calc_mag_field_symm()

bnd = 0.04761904761904756
p2 = PCB_u(2*M, None, c, orientation, x_bnd = [-1,1], y_bnd=[-1,1])
p2.assemble_S()
p2.calc_mag_field()

f = p2.field


# # check if the field on the bottom is the same
# orient = 2
# t1 = p.field_symm[-1,:,:,orient]

# t2 = p2.field[:,:,0, orient]

# diff = t1-t2

# plot the fields
scl = True
# plot_field_arrow_3d(p2.field, c, scaling=scl)
plot_field_arrow_3d(f, c, scaling=scl)

plot_field_arrow_3d(p.field_symm, c, scaling=scl)

plt.show()
