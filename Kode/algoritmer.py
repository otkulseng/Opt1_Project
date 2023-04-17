import numpy as np
from numpy.linalg import norm
from scipy.optimize import approx_fprime



def StrongWolfe(f, gradf, x,p,
                initial_value,
                initial_descent,
                initial_step_length = 1.0,
                c1 = 1e-2,
                c2 = 0.9,
                max_extrapolation_iterations = 50,
                max_interpolation_iterations = 20,
                rho = 2.0):
    '''
    Implementation of a bisection based bracketing method
    for the strong Wolfe conditions
    '''
    # initialise the bounds of the bracketing interval
    alphaR = initial_step_length
    alphaL = 0.0
    # Armijo condition and the two parts of the Wolfe condition
    # are implemented as Boolean variables
    next_x = x+alphaR*p
    next_value = f(next_x)
    next_grad = gradf(next_x)
    Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
    descentR = np.inner(p,next_grad)
    curvatureLow = (descentR >= c2*initial_descent)
    curvatureHigh = (descentR <= -c2*initial_descent)
    # We start by increasing alphaR as long as Armijo and curvatureHigh hold,
    # but curvatureLow fails (that is, alphaR is definitely too small).
    # Note that curvatureHigh is automatically satisfied if curvatureLow fails.
    # Thus we only need to check whether Armijo holds and curvatureLow fails.
    itnr = 0
    while (itnr < max_extrapolation_iterations and (Armijo and (not curvatureLow))):
        itnr += 1
        # alphaR is a new lower bound for the step length
        # the old upper bound alphaR needs to be replaced with a larger step length
        alphaL = alphaR
        alphaR *= rho
        # update function value and gradient
        next_x = x+alphaR*p
        next_value = f(next_x)
        next_grad = gradf(next_x)
        # update the Armijo and Wolfe conditions
        Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
        descentR = np.inner(p,next_grad)
        curvatureLow = (descentR >= c2*initial_descent)
        curvatureHigh = (descentR <= -c2*initial_descent)
    # at that point we should have a situation where alphaL is too small
    # and alphaR is either satisfactory or too large
    # (Unless we have stopped because we used too many iterations. There
    # are at the moment no exceptions raised if this is the case.)
    alpha = alphaR
    itnr = 0
    # Use bisection in order to find a step length alpha that satisfies
    # all conditions.
    while (itnr < max_interpolation_iterations and (not (Armijo and curvatureLow and curvatureHigh))):
        itnr += 1
        if (Armijo and (not curvatureLow)):
            # the step length alpha was still too small
            # replace the former lower bound with alpha
            alphaL = alpha
        else:
            # the step length alpha was too large
            # replace the upper bound with alpha
            alphaR = alpha
        # choose a new step length as the mean of the new bounds
        alpha = (alphaL+alphaR)/2
        # update function value and gradient
        next_x = x+alphaR*p
        next_value = f(next_x)
        next_grad = gradf(next_x)
        # update the Armijo and Wolfe conditions
        Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
        descentR = np.inner(p,next_grad)
        curvatureLow = (descentR >= c2*initial_descent)
        curvatureHigh = (descentR <= -c2*initial_descent)
    # return the next iterate as well as the function value and gradient there
    # (in order to save time in the outer iteration; we have had to do these
    # computations anyway)
    return alpha

def other_linesearch(pk,
               xk,
               f,
               gradf,
               alpha_max = 10,
               rho = 2,
               c1 = 1e-2,
               c2 = 0.9,
               Niter=100):
    return StrongWolfe(f, gradf, xk, pk, f(xk), np.dot(gradf(xk), pk))

def lineSearch(pk,
               xk,
               f,
               gradf,
               rho = 2,
               c1 = 1e-2,
               c2 = 0.9):

    Niter_extrapolate = 50
    Niter_interpolate = 20

    descent = np.dot(gradf(xk), pk)
    fxk = f(xk)

    alpha_high = 1
    alpha_low = 0

    # Strong wolfe conditions
    # 3.7a
    armijo = lambda alpha: f(xk + alpha * pk) <= fxk + c1 * alpha * descent

    # 3.7b
    curvatureLow = lambda alpha: c2 * descent <= np.dot(gradf(xk + alpha * pk), pk)
    curvatureHigh = lambda alpha: -c2 * descent >= np.dot(gradf(xk + alpha * pk), pk)

    for _ in range(Niter_extrapolate):
        if (not armijo(alpha_high)) or curvatureLow(alpha_high):
            break

        alpha_low = alpha_high
        alpha_high = rho * alpha_high

    alpha = (alpha_low + alpha_high)/2
    ar = armijo(alpha)
    cl = curvatureLow(alpha)
    ch = curvatureHigh(alpha)

    for _ in range(Niter_interpolate):
        if ar and cl and ch:
            break

        if ar and (not cl):
            alpha_low = alpha
        else:
            alpha_high = alpha
        alpha = (alpha_low + alpha_high)/2

        ar = armijo(alpha)
        cl = curvatureLow(alpha)
        ch = curvatureHigh(alpha)
    return alpha

def bfgs(x0, f, gradf=None, Niter=100, grad_epsilon=1e-12):
    if gradf is None:
        gradf = lambda xk : approx_fprime(xk, f, epsilon=grad_epsilon).astype('float64')

    x_current = x0
    x_next = np.copy(x0)
    grad_current = gradf(x0)


    Hk = np.identity(np.size(x0))
    I = np.identity(np.size(x0))

    n = 0

    sk = np.zeros((len(x_current), 1))
    yk = np.zeros((len(x_current), 1))

    while norm(grad_current) > grad_epsilon and n < Niter:
        n += 1
        pk = -Hk @ grad_current

        alpha = lineSearch(pk, x_current, f, gradf)

        x_next = x_current + alpha * pk
        grad_next = gradf(x_next)

        sk = x_next - x_current
        yk = grad_next - grad_current

        rhok = 1 / np.dot(yk, sk)

        if n==1:
            Hk = Hk*(1/(rhok*np.inner(yk,yk)))

        A = I - rhok * np.outer(sk, yk)
        B = rhok * np.outer(sk, sk)
        Hk = A @ Hk @ A.T
        Hk = Hk + B

        # z = Hk.dot(yk)
        # Hk += -rhok*(np.outer(sk,z) + np.outer(z,sk)) + rhok*(rhok*np.inner(yk,z)+1)*np.outer(sk,sk)

        x_current = x_next
        grad_current = grad_next


    if n == Niter:
        print("Maximum iteration obtained in BFGS method")
        print(f'Norm: {norm(grad_current)}')

    return x_current



