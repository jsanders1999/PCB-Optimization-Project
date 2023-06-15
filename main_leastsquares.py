
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
from Contour_new import *
from PlottingMethods_new import *

# plt.style.use("seaborn-v0_8-dark")

tar_dir = "z"

res = 25
M = 5
FACTOR = 1e5

plot_old = [1,0,1,0,0]

save_path = 0
# save_path = "C:/Users/aglas/OneDrive/Bureaublad/Documenten/Technische Wiskunde Master/Advanced Modelling/Project/Images/Final_"+ str(M)+"_"+tar_dir+"/"
save_path = "C:/Users/aglas/OneDrive/Bureaublad/Documenten/Technische Wiskunde Master/Advanced Modelling/Project/Images/"
cube = Cube(res)
pcb = PCB_u(M, None, cube, 0)
 
pcb.assemble_S()
pcb.assemble_A()
Q = pcb.SS_x + pcb.SS_y + pcb.SS_z
J = np.ones((pcb.cube.resolution**3,pcb.cube.resolution**3))
R = pcb.S_z.T@J@pcb.S_z

if tar_dir == "x":
    u = li.solve(Q,FACTOR*pcb.S_x.T@np.ones(res**3))
    Ju = np.concatenate((pcb.S_x@u - FACTOR*np.ones(res**3),pcb.S_y@u,pcb.S_z@u))
    V = (1/FACTOR)**2*(1/res**3)*np.linalg.norm(Ju)**2
    B = 1/FACTOR*1/res**3*np.sum(pcb.S_x@u)

if tar_dir == "z":
    u = li.solve(Q,pcb.S_z.T@np.ones(res**3))
    Ju = np.concatenate((pcb.S_x@u,pcb.S_y@u,pcb.S_z@u- FACTOR*np.ones(res**3)))
    V = (1/FACTOR)**2*(1/res**3)*np.linalg.norm(Ju)**2
    B = 1/FACTOR*1/res**3*np.sum(pcb.S_z@u)

# B_ls = [1/res**3*np.sum(pcb.S_x@u),1/res**3*np.sum(pcb.S_y@u),1/res**3*np.sum(pcb.S_z@u)]
# B_mean = 1/FACTOR*np.mean(B_ls)*3

P = 1e-7*u.T@pcb.A@u
print(np.max(u),np.min(u))

print("Uniformity: ", V)
print("Magnetic Field: ", B)
print("Power: ", P)

pcb.u_cart = np.reshape(u, (M,M), order = "C")



if sum(plot_old) != 0:
    plot_other(pcb,u,save_path,V,B,toplot = plot_old)

# contours, I = CalcContoursHat(pcb, tar_dir,seperation = 0.003,savepath = save_path, plot = True)
# contours, I = CalcContoursHat(pcb, tar_dir,seperation = 0.001,savepath = save_path, plot = True)
# contours, I = CalcContoursHat(pcb, tar_dir,seperation = 0.01,savepath = save_path, plot = True)
# contours, I = CalcContoursHat(pcb, tar_dir,seperation = 0.0003,savepath = save_path, plot = True)

# Bx, By, Bz, U = CalcUniformityContours(contours, res, I, V = [-0.5, 0.5, -0.5, 0.5, 0.1, 1.1])
# pcb.system_analysis()
# print(pcb.condition)
# MultiPlot(4, 16,FACTOR,tar_dir, res = 25,save_path =save_path)
# MultiPlot_contour(4, 7, tar_dir, seperation = 0.0003, f = 1e5, res =25,save_path = 0)
# UnifMagFieldPower_xz(4, 16, f = FACTOR, res = 10,save_path = save_path)
# UnifMagFieldPower_OptivsWired(8,9,"z",seprs = [0.0025,0.005], res = 14,save_path = 0)
plt.show()


