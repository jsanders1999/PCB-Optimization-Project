# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:17:36 2023

@author: jeroe
"""

import numpy as np
import matplotlib.pyplot as plt

from hatfunctions import current_lambda, lambda_1D, dlambda_1D, lambda_2D
from PlottingFunctions import figsize, figsize_square, MEDIUM_FONT

plt.close('all')
x_min = -1
x_max = 1
N = 100
x_0 = 1

x = np.linspace(x_min, x_max, N)
y = np.linspace(x_min, x_max, N)

X, Y = np.meshgrid(x,y)

curr_x, curr_y = current_lambda(X,Y,x_0)

fig, ax = plt.subplots(1,1, figsize = figsize)


# plot 2D hat pontential
pc = ax.pcolormesh(X,Y,lambda_2D(X, Y, x_0))
plt.colorbar(pc)

# plot quiver of current
N = 10
x = np.linspace(x_min, x_max, N)
y = np.linspace(x_min, x_max, N)
X, Y = np.meshgrid(x,y)

curr_x, curr_y = current_lambda(X,Y,x_0)

ax.quiver(X, Y, curr_x, curr_y)

ax.set_xlabel("x", fontsize=MEDIUM_FONT)
ax.set_ylabel("y", fontsize=MEDIUM_FONT)

plt.savefig("plot_hat_function.png", bbox_inches = 'tight', pad_inches=0.1)