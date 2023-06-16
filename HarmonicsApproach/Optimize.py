
import numpy as np
import scipy as sp
from numba import njit

@njit
def Uniformity(u, Q, R, N):
    """ The function that is minimized. Is a measure for the uniformity of the magnetic field in the target volume"""
    return (u.T@Q@u)/(u.T@(R/N**3)@u) - 1

@njit
def GradUniformity(u, Q, R, N):
    """ The gradient of Uniformity() w.r.t u"""
    u = u/np.linalg.norm(u)
    Ru = (R/N**3)@u
    Qu = Q@u
    uRu = u.T@Ru
    uQu = u.T@Qu
    return 2*(Qu-(uQu/uRu)*Ru)/uRu

@njit
def HessUniformity(u, Q, R, N, M):
    """ The gradient of Uniformity() w.r.t u"""
    #Not working
    Ru = R@u
    Qu = Q@u
    uRu = u.T@Ru
    uQu = u.T@Qu
    I = np.eye(M)
    return 2*N**3*(Q/uRu - uQu/uRu**2*R - 2/uRu**2 * Qu@Ru*I + 4*uQu/uRu**3*Qu@Ru*I - 2*Qu@Qu/uRu**2* I )


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

@njit
def bfgs_minimize_uniformity(Q, R, N, u0, max_iterations=100, tolerance=1e-6):
    """
    WRITTEN BY CHATGPT, DOESN"T WORK!!!
    Perform BFGS optimization to minimize the uniformity function. Does not seem to work yet.

    Args:
        Q (numpy.ndarray): Q matrix.
        R (numpy.ndarray): R matrix.
        N (int): Dimensionality.
        u0 (numpy.ndarray): Initial guess for u.
        max_iterations (int): Maximum number of iterations (default: 100).
        tolerance (float): Convergence tolerance (default: 1e-6).

    Returns:
        numpy.ndarray: Optimized solution for u.
    """
    # Initialize variables
    u = u0.copy()
    H = np.eye(len(u0))  # Initial approximation of the Hessian matrix

    for iteration in range(max_iterations):
        # Compute gradient and objective function value
        gradient = GradUniformity(u, Q, R, N)
        objective = Uniformity(u, Q, R, N)

        # Check convergence
        if np.linalg.norm(gradient) < tolerance:
            break

        # Update search direction using the inverse Hessian approximation
        direction = -H@gradient #-np.linalg.solve(H, gradient)

        # Perform line search to find the step size
        step_size = 1.0
        while Uniformity(u + step_size * direction, Q, R, N) >= objective:
            step_size *= 0.5

        # Update parameters
        u_new = u + step_size * direction
        gradient_new = GradUniformity(u_new, Q, R, N)

        # Update Hessian approximation using BFGS formula
        y = gradient_new - gradient
        s = u_new - u
        Hy = H @ y
        sHy = np.dot(s, Hy)
        if sHy > 0:
            H = H - np.outer(s, Hy) / sHy - np.outer(Hy, s) / sHy + (sHy + np.dot(y, Hy)) * np.outer(s, s) / (sHy ** 2)

        # Update u for the next iteration
        u = u_new

    return u

def bfgs_optimization(f, grad_f, x0, max_iter=100, tol=1e-6):
    """
    ONLY WORKS UP TILL 7 HARMONICS FOR SOME REASON
    BFGS optimization algorithm for finding the minimum of a function.

    Args:
        f (callable): The objective function to minimize.
        grad_f (callable): The gradient function of the objective function.
        x0 (numpy.ndarray): Initial guess for the optimization.
        max_iter (int): Maximum number of iterations. Default is 100.
        tol (float): Tolerance for convergence. Default is 1e-6.

    Returns:
        tuple: A tuple containing the optimized solution and the value of the objective function at the solution.
    """
    n = len(x0)
    H = np.eye(n)  # Initial approximation of the inverse Hessian
    x = x0.copy()
    f_x = f(x)
    grad_x = grad_f(x)
    iter_count = 0

    while np.linalg.norm(grad_x) > tol and iter_count < max_iter:
        p = -H @ grad_x  # Search direction

        # Line search to determine step size
        alpha = backtracking_line_search(f, grad_f, x, p)

        x_new = x + alpha * p
        s = x_new - x
        grad_new = grad_f(x_new)
        y = grad_new - grad_x

        # BFGS update of the inverse Hessian approximation
        rho = 1 / (y @ s)
        H = (np.eye(n) - rho * np.outer(s, y)) @ H @ (np.eye(n) - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x = x_new/np.linalg.norm(x_new)
        f_x = f(x)
        grad_x = grad_new
        iter_count += 1

    return x, f_x

def backtracking_line_search(f, grad_f, x, p, alpha=0.3, beta=0.8):
    """
    Backtracking line search to determine the step size in the optimization. Used in BFGS Method

    Args:
        f (callable): The objective function.
        grad_f (callable): The gradient function of the objective function.
        x (numpy.ndarray): Current solution.
        p (numpy.ndarray): Search direction.
        alpha (float): Scaling factor for the sufficient decrease condition. Default is 0.3.
        beta (float): Reduction factor for the step size. Default is 0.8.

    Returns:
        float: Step size satisfying the Armijo condition.
    """
    t = 1.0
    while f(x + t * p) > f(x) + alpha * t * grad_f(x).dot(p):
        t *= beta
    return t

