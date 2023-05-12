
import numpy as np
import scipy as sp
from numba import njit

@njit
def Uniformity(u, Q, R, N):
    """ The function that is minimized. Is a measure for the uniformity of the magnetic field in the target volume"""
    return (N**3*u.T@Q@u)/(u.T@R@u) - 1

@njit
def GradUniformity(u, Q, R, N):
    """ The gradient of Uniformity() w.r.t u"""
    Ru = R@u
    Qu = Q@u
    uRu = u.T@Ru
    uQu = u.T@Qu
    return 2*N**3*(Qu-(uQu/uRu)*Ru)/uRu

@njit
def HessUniformity(u, Q, R, N, M):
    """ The gradient of Uniformity() w.r.t u"""
    Ru = R@u
    Qu = Q@u
    uRu = u.T@Ru
    uQu = u.T@Qu
    I = np.eye(M, M)
    return 2*N**3*(Q/uRu - uQu/uRu^2*R - 2/uRu^2 * Qu@Ru*  I + 4*uQu/uRu^3*Qu@Ru*I - 2*Qu@Qu/uRu^2* I )


def line_search_line_min(Q, R, res, u_norm, u_start, n_steps, alpha = 0.5):
    N = res
    u_old = u_start
    Vs = [Uniformity(u_old, Q, R, N)]
    #improv = Uniformity(u_old, Q, R, N)
    count = 0
    u_best = u_old
    V_best = Vs[-1]
    step = 0.0
    while count < n_steps:# and improv>1:
        grad = GradUniformity(u_old, Q, R, N)
        u_normed = u_old/np.linalg.norm(u_old)
        d = grad -  (u_normed.T @ grad)*u_normed  #project the gradient onto the tangent plane of the contraint sphere
        #print("d :", d)
        if True: #d.T@self.R@d * u_old.T@self.Q@d > d.T@self.Q@d * u_old.T@self.R@d:
            func = lambda t: Uniformity(u_old - t*d, Q, R, N )
            jac = lambda t: -GradUniformity(u_old - t*d, Q, R, N).T@d
            optimal_res = sp.optimize.minimize(func, np.random.normal(), jac = jac) #np.random.normal()
            t_optimal = optimal_res.x[0]
            if optimal_res.success == False:
                print("No minimum found in line search")#raise ValueError
            #print("t :", t_optimal)
            #t_optimal2 = -1/2*(d.T@Q@d * u_old.T@R@u_old - d.T@R@d * u_old.T@Q@u_old)/(d.T@R@d * u_old.T@Q@d - d.T@Q@d * u_old.T@R@d)
            #print(t_optimal, t_optimal2)
            step = alpha*step - t_optimal*d
            u_new = u_old + step
        else:
            print(count, self.opti_func(u_old), "No minimium found in line search")            
            u_new = u_old + 1e-5*np.linalg.norm(u_old)*np.random.normal(size = u_old.size) #random perturbation
        u_new = ((u_new)/np.linalg.norm(u_new))*u_norm
        V_new = Uniformity(u_new, Q, R, N)
        V_old = Uniformity(u_old, Q, R, N)
        improv = V_old - V_new 
        #print("u_old : ", Uniformity(u_old, Q, R, N) )
        #print("u_new : ", Uniformity(u_new, Q, R, N))
        #print(improv)
        #if improv>0:
        u_old = u_new
        #else:
        #    print("no improvement!")
        #    u_old +=  u_norm*1e-4/np.sqrt(count/10000+1)*np.linalg.norm(u_old)*np.random.normal(size = u_old.size)
        #    u_old = u_old/np.linalg.norm(u_old)*u_norm
        #    V_old = Uniformity(u_old, Q, R, N)
        Vs.append(V_old)
        if Vs[-1] <= V_best:
            u_best = u_old
            V_best = Vs[-1]
        count +=1
        if count%100 ==0:
            print(count, " : ", Vs[-1])
    return u_old, Vs[-1], u_best, V_best