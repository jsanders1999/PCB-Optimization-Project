
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
    return N**3*(Qu-(uQu/uRu)*Ru)/uRu


def cartesian_to_sphere(u):
    """Transforms cartesian vector u to spherical vector phi
    first entry of phi is the length of u, rest is angle coordinates
    """
    phi = np.zeros(len(u)-1)
    r = np.linalg.norm(u)
    for i in range(len(u)-1):
        phi[i] = np.arccos(u[i]/(np.sqrt(sum(u[i]**2 for i in range(i,len(u))))))
    if u[-1] >=0:
        phi[-1] = np.arccos(u[-1]/np.sqrt(u[-1]**2+u[-2]**2))
    else:
        phi[-1] = 2*np.pi-np.arccos(u[-1]/np.sqrt(u[-1]**2+u[-2]**2))
    return r,phi


def sphere_to_cartesian(r,phi):
    """Transforms spherical vector phi to cartesian vector u"""
    u = np.zeros(len(phi)+1)
    for i in range(0,len(phi)):
        u[i] = r
        for j in range(0,i):
            u[i]*= np.sin(phi[j])
        u[i] *= np.cos(phi[i])
    u[-1] = r
    for j in range(0,len(phi)):
        u[-1]*= np.sin(phi[j])
    return u


def GradUniformity_spherical(r,phi, Q, R, N):
    """ The gradient of Uniformity() for spherical coordinates"""
    u = sphere_to_cartesian(r,phi)
    grad = GradUniformity(u, Q, R, N)
    grad_phi = np.zeros(len(phi))
    # for i in range(len(u)):
    #     dxidr = 1
    #     for j in range(i):
    #         dxidr*=np.sin(phi[j])
    #     if i != len(u)-1:
    #         dxidr*=np.cos(phi[i])
    #     else:
    #         dxidr*=np.sin(phi[i])
    #     grad_phi[0] += dxidr*grad[i]
    for k in range(len(grad_phi)):
        dxkdpk = r*-np.sin(phi[k])
        for j in range(k):
            dxkdpk*=np.sin(phi[j])
        grad_phi[k] += dxkdpk*grad[k]
        for i in range(k+1,len(u)-1):
            dxidpk = r
            for j in range(i):
                if j !=k:
                    dxidpk*=np.sin(phi[j])
                else:
                    dxidpk*=np.cos(phi[j])
            if i != len(u)-1:
                dxidpk*=np.cos(phi[i])
            else:
                dxidpk*=np.sin(phi[i])
            grad_phi[k] += dxidpk*grad[i]
    return grad_phi

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
                if step%1000 ==0:# or step <50:
                    print("itteration: ", step, "Uniformity:", Uniformity(u, Q, R, N))
        return u, Uniformity(u, Q, R, N)

class optimize_k:
    def __init__(self, pcb, verbose = True):
        self.pcb = pcb
        self.verbose = verbose
        self.Q = pcb.SS_x + pcb.SS_y + pcb.SS_z
        self.J = np.ones((pcb.cube.resolution**3,pcb.cube.resolution**3))
        self.R = pcb.S_z.T@self.J@pcb.S_z #+ pcb.S_x.T@self.J@pcb.S_x + pcb.S_y.T@self.J@pcb.S_y + 
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
    
    
    def opti_grad_sphere(self,r,phi):
        return GradUniformity_spherical(r,phi, self.Q, self.R, self.pcb.cube.resolution)

             
    
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
    
    def line_search_line_min(self, u_start, n_steps, u_norm, alpha = 0.5):
        u_old = u_start
        Vs = [self.opti_func(u_old)]
        improv = self.opti_func(u_old)
        count = 0
        # print(count, " : ", Vs[-1])
        while count < n_steps:# and improv>1:
            grad = self.opti_grad(u_old)
            u_normed = u_old/np.linalg.norm(u_old)
            d = grad -  (u_normed.T @ grad)*u_normed  #project the gradient onto the tangent plane of the contraint sphere
            if True: #d.T@self.R@d * u_old.T@self.Q@d > d.T@self.Q@d * u_old.T@self.R@d:
                func = lambda t: self.opti_func(u_old - t*d)
                t_optimal = sp.optimize.minimize(func, 5e4*np.random.normal()).x[0]
                # print(t_optimal)
                #t_optimal = -1/2*(d.T@self.Q@d * u_old.T@self.R@u_old - d.T@self.R@d * u_old.T@self.Q@u_old)/(d.T@self.R@d * u_old.T@self.Q@d - d.T@self.Q@d * u_old.T@self.R@d)
                step = - t_optimal*d #+alpha*step 
                u_new = u_old + step
                #print(self.opti_func(u_old), self.opti_func(u_new))
                #print(func(0), func(t_optimal))
            else:
                print(count, self.opti_func(u_old), "No minimium found in line search")            
                u_new = u_old + 1e-5*np.linalg.norm(u_old)*np.random.normal(size = u_old.size) #random perturbation
            u_new = u_norm*((u_new)/np.linalg.norm(u_new))
            improv = Vs[-1] - self.opti_func(u_new)
            u_old = u_new
            Vs.append(self.opti_func(u_old))
            count +=1
            if self.verbose:
                if count%100 == 0:
                    print(count, " : ", Vs[-1])
        return u_old, Vs
    
    def scipy_minimum(self, u_start, u_norm):
        optres = sp.optimize.minimize(self.opti_func, u_start, jac = self.opti_grad )
        print(optres)
        u = optres.x/np.linalg.norm(optres.x)*u_norm
        V = self.opti_func(optres.x)
        return u, V


    def line_search_line_perturb(self, u_start, n_steps, u_norm):
        u = u_start
        Vs = [self.opti_func(u)]
        improv = self.opti_func(u)
        count = 0
        while count < n_steps:# and improv>1:
            grad = self.opti_grad(u)
            u_normed = u/np.linalg.norm(u)
            d = grad -  (u_normed @ grad)*u_normed 
            if d@self.R@d*u@self.Q@d > d@self.Q@d*u@self.R@d and improv > 1:
                t_optimal = 1/2*(d@self.Q@d*u@self.R@u - d@self.R@d*u@self.Q@u)/(d@self.R@d*u@self.Q@d-d@self.Q@d*u@self.R@d)
                u += t_optimal*d
            elif d@self.R@d*u@self.Q@d > d@self.Q@d*u@self.R@d:
                print(count,self.opti_func(u), "perturb u randomly")            
                random_pert = np.random.rand(len(u))
                u = u + 0.1*(np.linalg.norm(u)/np.linalg.norm(random_pert))*random_pert
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
    
    def line_search_sphere(self, u_start, n_steps, u_norm, alpha = 0.5):
        r,phi_old = cartesian_to_sphere(u_start)
        Vs = [self.opti_func(u_start)]
        improv = self.opti_func(u_start)
        count = 0
        while count < n_steps:# and improv>1:
            grad_phi = self.opti_grad_sphere(r,phi_old)
            # if True: #d.T@self.R@d * u_old.T@self.Q@d > d.T@self.Q@d * u_old.T@self.R@d:
            func = lambda t: self.opti_func(sphere_to_cartesian(r,phi_old - t*grad_phi))
            t_optimal = sp.optimize.minimize(func, np.random.normal()).x[0]
            #t_optimal = -1/2*(d.T@self.Q@d * u_old.T@self.R@u_old - d.T@self.R@d * u_old.T@self.Q@u_old)/(d.T@self.R@d * u_old.T@self.Q@d - d.T@self.Q@d * u_old.T@self.R@d)
            step = -t_optimal*grad_phi #+alpha*step
            phi_new = phi_old + step
            # else:
            #     print(count, self.opti_func(u_old), "No minimium found in line search")            
            #     u_new = u_old + 1e-5*np.linalg.norm(u_old)*np.random.normal(size = u_old.size) #random perturbation
            Vs.append(self.opti_func(sphere_to_cartesian(r,phi_new)))
            improv = Vs[-2] - Vs[-1]
            phi_old = phi_new
            count +=1
            if self.verbose:
                print(count, " : ", Vs[-1])
        return sphere_to_cartesian(r,phi_old), Vs