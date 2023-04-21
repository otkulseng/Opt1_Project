import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np


class TensegrityStructure:
    def __init__(self, fixed_points, free_weights, cables, bars, k=0, c=0, rho=0, quadratic_penalization=False, solution=None) -> None:
        self.fixed_points = fixed_points
        self.free_weights = free_weights
        self.cables = cables
        self.bars = bars

        self.func = gen_E(cables, bars, free_weights,
                          fixed_points, k, c, rho, quadratic_penalization)
        self.grad = gen_grad_E(cables, bars, free_weights,
                               fixed_points, k, c, rho, quadratic_penalization)

        self.solution = solution

    def diffToSolution(self, sol):
        if self.solution is None:
            print("No solution for this structure is given")
            return -1
        sol = np.reshape(sol, (-1, 3))
        self.solution = np.reshape(self.solution, (-1, 3))
        return np.linalg.norm(sol - self.solution)

    def __plot(self, ax, sol, angle=None,title=""):
        x = np.reshape(sol, (-1, 3))
        p = self.fixed_points
        if len(p) > 0:
            points = np.concatenate((p, x))
        else:
            points = x

        for i in range(len(p)):
            ax.scatter(p[i][0], p[i][1], p[i][2], c='k', s=200)
            ax.text(p[i][0], p[i][1], p[i][2],
                    r'$p_{%d}$' % i, size=20, zorder=1, color='k')

        for i in range(len(x)):
            ax.scatter(x[i][0], x[i][1], x[i][2], c='g', s=200)
            ax.text(x[i][0], x[i][1], x[i][2],
                    r'$x_{%d}$' % i, size=20, zorder=1, color='k')

        for elem in self.cables:
            a = elem[0]
            b = elem[1]
            cable = np.array([points[a], points[b]]).T
            ax.plot(cable[0], cable[1], cable[2], c='k',
                    linestyle='dashed', linewidth=3)

        for elem in self.bars:
            a = elem[0]
            b = elem[1]
            bar = np.array([points[a], points[b]]).T
            ax.plot(bar[0], bar[1], bar[2], c='k', linewidth=3)

        if angle is not None:
            elev, azim = angle
            #elev, azim = angle
            ax.view_init(elev=elev, azim=azim)
            #ax.view_init(elev=90, azim=0)
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        ax.set_zlabel("z")
        ax.set_title(title)
        #ax.set_title("Automatisert tittel?")

    def plot(self, sol, x0 = None):

        #fig = plt.figure(figsize=(10, 30))
        #ax = plt.axes(projection = '3d')
        fig, axes = plt.subplots(nrows=1, ncols=3, subplot_kw={
                                 "projection": "3d"}, figsize=(10, 30))


        ax1, ax2, ax3 = axes
        if x0 is not None:
            self.__plot(ax1, x0,  (25, 10),title="Initialisation")
        else:
            self.__plot(ax1, sol,  (25, 10))
        
        self.__plot(ax2, sol, (25, 10),title="Solution")
        self.__plot(ax3, sol, (90, 5),title="Solution seen from above")

        return fig, axes


def cable_elastic(state, cables, k):
    l = state[cables[:, 0]]
    r = state[cables[:, 1]]
    diff = np.linalg.norm(l - r, axis=1) - cables[:, 2]
    diff = np.where(diff < 0, 0, diff)
    return (k / (2*cables[:, 2]**2) * diff) @ diff


def bar_elastic(state, bars, c):
    l = state[bars[:, 0]]
    r = state[bars[:, 1]]
    diff = np.linalg.norm(l - r, axis=1) - bars[:, 2]

    return (c / (2*bars[:, 2]**2) * diff) @ diff


def bar_potential(state, bars, rho):
    l = state[bars[:, 0]]
    r = state[bars[:, 1]]
    return 0.5 * rho * bars[:, 2] @ (l[:, 2] + r[:, 2])


def gen_E(cables, bars, free_weights, fixed_points, k, c, rho, quadratic_penalization=False):
    """Takes in the position of all fixed points p,
    and information about all the cables and bars. Generates the
    objective function used with E_cable_elast

    Returns:
        objective function f(x) where x is a numpy array of free
        optimization variables
    """
    if quadratic_penalization:
        def func(mu_input):
            orig = gen_E(cables, bars, free_weights, fixed_points,
                         k, c, rho, quadratic_penalization=False)

            def inner(xx):
                x = np.reshape(xx, (-1, 3))

                # E = 0
                # for  i in range(len(x)):
                #     z = x[i][2]
                #     curVal = max(-z, 0)
                #     E += curVal**2

                E = np.where(x[:, 2] < 0, x[:, 2], 0)
                return orig(xx) + 0.5 * mu_input * E @ E

            def without_fixed(xx):
                E0 = 0.5 * mu_input * (xx[0]**2 + xx[1] ** 2)
                E1 = 0.5 * mu_input * (xx[0] - xx[3])**2
                return inner(xx) + E0 + E1

            if len(fixed_points) == 0:
                return without_fixed

            return inner

        return func

    state = np.zeros((len(fixed_points) + len(free_weights), 3))
    if len(fixed_points) > 0:
        state[:len(fixed_points)] = fixed_points

    def func(xx):
        x = np.reshape(xx, (-1, 3))
        state[len(fixed_points):] = x

        E_cable_elastic = cable_elastic(state, cables, k)
        E_bar_elastic = bar_elastic(state, bars, c)
        E_bar_potential = bar_potential(state, bars, rho)
        E_ext = free_weights @ x[:, 2]
        return E_cable_elastic + E_bar_elastic + E_bar_potential + E_ext

    def func_without_bars(xx):
        x = np.reshape(xx, (-1, 3))
        state[len(fixed_points):] = x
        E_cable_elastic = cable_elastic(state, cables, k)
        E_ext = free_weights @ x[:, 2]

        return E_cable_elastic + E_ext

    def func_without_cables(xx):
        x = np.reshape(xx, (-1, 3))
        state[len(fixed_points):] = x

        E_bar_elastic = bar_elastic(state, bars, c)
        E_bar_potential = bar_potential(state, bars, rho)
        E_ext = free_weights @ x[:, 2]
        return E_bar_elastic + E_bar_potential + E_ext

    if len(bars) == 0:
        return func_without_bars

    if len(cables) == 0:
        return func_without_cables

    return func


def gen_grad_E(cables, bars, free_weights, fixed_points, k, c, rho, quadratic_penalization=False):
    vec = np.array([0, 0, 1])

    if quadratic_penalization:
        def func(mu):
            orig = gen_grad_E(cables, bars, free_weights, fixed_points,
                              k, c, rho, quadratic_penalization=False)

            def inner(xx):
                x = np.reshape(xx, (-1, 3))
                grad = np.zeros(x.shape)

                for i in range(len(x)):
                    z = max(-x[i][2], 0)
                    grad[i] -= mu * z * vec

                return orig(x) + grad.flatten()

            def without_fixed(xx):
                x = np.reshape(xx, (-1, 3))
                grad = np.zeros(x.shape)

                for i in range(len(x)):
                    z = max(-x[i][2], 0)
                    grad[i] -= mu * z * vec

                # equality constraint to origo
                grad[0] += mu * np.array([x[0][0], x[0][1], 0])

                diff = mu * (xx[0] - xx[3]) * np.array([1, 0, 0])
                grad[0] += diff
                grad[1] -= diff

                return orig(x) + grad.flatten()

            if len(fixed_points) == 0:
                return without_fixed
            return inner
        return func

    state = np.zeros((len(fixed_points) + len(free_weights), 3))
    if len(fixed_points) > 0:
        state[:len(fixed_points)] = fixed_points

    def func(xx):
        x = np.reshape(xx, (-1, 3))
        state[len(fixed_points):] = x
        grad = np.zeros(x.shape)

        # free weights
        for i in range(len(grad)):
            grad[i][2] += free_weights[i]
        # grad[:, 2] = grad[:, 2] + free_weights

        # cables
        for [i, j, lij] in cables:
            xl = state[i]
            xr = state[j]

            temp_norm = np.linalg.norm(xl - xr) + 1e-20
            num = k / lij**2 * (1 - lij / temp_norm)
            num = max(num, 0)  # 0 if lij / norm < 1
            elem = num * (xl - xr)

            if i >= len(fixed_points):
                idx = i - len(fixed_points)
                grad[idx] = grad[idx] + elem
            if j >= len(fixed_points):
                idx = j - len(fixed_points)
                grad[idx] = grad[idx] - elem
        # end cables

        # bar potential
        for [i, j, lij] in bars:
            if i >= len(fixed_points):
                i = i - len(fixed_points)
                grad[i] += 0.5 * rho * lij * vec
            if j >= len(fixed_points):
                j = j - len(fixed_points)
                grad[j] += 0.5 * rho * lij * vec
        # end bar potential

        # bar elastic
        for [i, j, lij] in bars:
            xl = state[i]
            xr = state[j]

            temp_norm = np.linalg.norm(xl - xr) + 1e-20
            num = c / lij**2 * (1 - lij / temp_norm)
            elem = num * (xl - xr)
            if i >= len(fixed_points):
                idx = i - len(fixed_points)
                grad[idx] = grad[idx] + elem
            if j >= len(fixed_points):
                idx = j - len(fixed_points)
                grad[idx] = grad[idx] - elem
        # end bar elastic

        return grad.flatten()

    return func
