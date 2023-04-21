import numpy as np
from scipy.optimize import approx_fprime

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

def bfgs(x0, f, gradf=None, Niter=100, grad_epsilon=1e-10, plot_summary=False, convergence_plot=False, return_iteration=False):
    if gradf is None:
        gradf = lambda xk : approx_fprime(xk, f, epsilon=grad_epsilon)


    x_current = x0
    x_next = np.copy(x0)
    grad_current = gradf(x0)
    norm = np.linalg.norm(grad_current)

    if convergence_plot:
        convergence = np.zeros(Niter + 1)
        convergence[0] = norm

    Hk = np.identity(np.size(x0))
    I = np.identity(np.size(x0))

    n = 0

    sk = np.zeros((len(x_current), 1))
    yk = np.zeros((len(x_current), 1))

    while norm > grad_epsilon and n < Niter:
        n += 1
        pk = -Hk @ grad_current

        alpha = lineSearch(pk, x_current, f, gradf)

        x_next = x_current + alpha * pk
        grad_next = gradf(x_next)

        sk = x_next - x_current
        yk = grad_next - grad_current

        tempdot = np.dot(yk, sk)
        if tempdot == 0:
            break;
        rhok = 1 / tempdot

        if n==1:
            Hk = Hk*(1/(rhok*np.inner(yk,yk)))

        A = I - rhok * np.outer(sk, yk)
        Hk = A @ Hk @ A.T + rhok * np.outer(sk, sk)

        x_current = x_next
        grad_current = grad_next
        norm = np.linalg.norm(grad_current)

        if convergence_plot:
            convergence[n] = norm

    if plot_summary:
        if n == Niter:
            print("Maximum iteration obtained in BFGS method")
        else:
            print("BFGS converged!")


        print(f'Gradient at solution: \n {np.reshape(grad_current, (-1, 3))} \n with norm: \n{np.linalg.norm(grad_current)}\n')
        print(f'Solution: \n{np.reshape(x_current, (-1, 3))}\n with function value: \n{f(x_current)}\n')

    if convergence_plot:
        return x_current, convergence[:n]

    if return_iteration:
        return x_current, n

    return x_current



