
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

@njit
def Uniformity(u, Q, R, N):
    """ The function that is minimized. Is a measure for the uniformity of the magnetic field in tha target volume"""
    return (N**3*u.T@Q@u)/(u.T@R@u) - 1

@njit
def GradUniformity(u, Q, R, N):
    """ The gradient of Uniformity() w.r.t u"""
    Ru = R@u
    Qu = Q@u
    uRu = u.T@Ru
    uQu = u.T@Qu
    return (N**3*Qu-(uQu/uRu)*Ru)/uRu

def UniformityDirected():
    """ The function that is minimized. Is a measure for the uniformity of the magnetic field in the target volume for a specific direction"""
    return

def GradUniformityDirected():
    """ The gradient of UniformityDirected() w.r.t u"""
    return

@njit
def gradient_descent_normed(u_start, Q, R, N,  n_steps, stepsize, u_norm):
        u = u_start/np.linalg.norm(u_start)*u_norm 
        for step in range(n_steps):
            grad = GradUniformity(u, Q, R, N)
            u += -grad*(stepsize/(np.sqrt(np.sqrt(step))+5))
            u = u/np.linalg.norm(u)*u_norm
            if True:
                print("itteration: ", step, "Uniformity:", Uniformity(u, Q, R, N))
        return u, Uniformity(u, Q, R, N)

class optimize_k:
    def __init__(self, pcb, verbose = True):
        self.pcb = pcb
        self.verbose = verbose
        self.Q = pcb.SS_x + pcb.SS_y + pcb.SS_z
        self.J = np.ones((pcb.cube.resolution**3,pcb.cube.resolution**3))
        self.R = pcb.S_x.T@self.J@pcb.S_x + pcb.S_y.T@self.J@pcb.S_y + pcb.S_z.T@self.J@pcb.S_z
        #self.L = 1/pcb.M*(np.diag(np.ones(pcb.M)*1/3) + np.diag(np.ones(pcb.M-1)*-1/12,k=1) +np.diag(np.ones(pcb.M-1)*-1/12,k=-1))
        #self.A = np.kron(np.eye(pcb.M),self.L)+np.kron(self.L,np.eye(pcb.M))
        if self.verbose:
            print("Q dims:", self.Q.shape)
            print("J dims", self.J.shape)
            print("R dims", self.R.shape)

    def opti_func(self,u):
        return Uniformity(u, self.Q, self.R, self.pcb.cube.resolution)

    def opti_grad(self, u):
        return GradUniformity(u, self.Q, self.R, self.pcb.cube.resolution)
             
    
    def line_search_normed(self,u_start,n_steps,line_step,eps,u_norm):
        u = u_start
        grad = self.opti_grad(u_start)
        us = [u]
        grads = [grad]
        Vs = [self.opti_func(u)]
        improv = self.opti_func(u)
        while improv > 0.01:
            grad = self.opti_grad(u)
            a = u
            b = u - line_step*self.opti_grad(u)
            count = 0
            while self.opti_func(u - (b+1)*line_step*self.opti_grad(u)) < self.opti_func(u - b*line_step*self.opti_grad(u)):
                b-= line_step*self.opti_grad(u)
            #print(np.linalg.norm(a-b))
            while np.linalg.norm(a-b) > eps*line_step:
                m = (a+b)/2
                if np.linalg.norm((m-line_step*self.opti_grad(m)) - b) < np.linalg.norm((m-line_step*self.opti_grad(m)) -a):
                    a = m
                else:
                    b = m
            u = ((u_norm)/np.linalg.norm((a+b)/2))*(a+b)/2
            improv = abs(Vs[-1] - self.opti_func(u))
            us.append(u)
            grads.append(grad)
            Vs.append(self.opti_func(u))
            print(count, improv)
        return us, Vs,grads 
    
    def gradient_descent_normed(self, u_start, n_steps, stepsize, u_norm):
        return gradient_descent_normed(u_start, self.Q, self.R, self.pcb.cube.resolution,  n_steps, stepsize, u_norm)
    
    def line_search_line_min(self, u_start, n_steps, u_norm):
        u = u_start
        Vs = [self.opti_func(u)]
        improv = self.opti_func(u)
        count = 0
        while count < n_steps:# and improv>1:
            grad = self.opti_grad(u)
            u_normed = u/np.linalg.norm(u)
            d = grad -  (u_normed @ grad)*u_normed 
            if d@self.R@d*u@self.Q@d > d@self.Q@d*u@self.R@d:
                t_optimal = 1/2*(d@self.Q@d*u@self.R@u - d@self.R@d*u@self.Q@u)/(d@self.R@d*u@self.Q@d-d@self.Q@d*u@self.R@d)
                u += t_optimal*d
            else:
                print(count, self.opti_func(u), "?? D vector scaled to sphere")            
                u = -d
            u = ((u_norm)/np.linalg.norm(u))*u
            improv = abs(Vs[-1] - self.opti_func(u))
            Vs.append(self.opti_func(u))
            count +=1
            if self.verbose:
                print(count, " : ", Vs[-1])
        return u, Vs