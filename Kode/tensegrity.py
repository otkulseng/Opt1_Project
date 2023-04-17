import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

class TensegrityStructure:
    def __init__(self, fixed_points, free_weights, cables, bars, k=0, c=0, rho=0, solution = None) -> None:
        self.fixed_points = fixed_points
        self.free_weights = free_weights
        self.cables = cables
        self.bars = bars

        self.func = gen_E(cables, bars, free_weights, fixed_points, k, c, rho)
        self.grad = gen_grad_E(cables, bars, free_weights, fixed_points, k, c, rho)

        self.solution = solution

    def diffToSolution(self, sol):
        if self.solution is None:
            print("No solution for this structure is given")
            return -1
        sol = np.reshape(sol, (-1, 3))
        self.solution = np.reshape(self.solution, (-1, 3))
        return np.linalg.norm(sol - self.solution)

    def plot(self, sol):
        x = np.reshape(sol, (-1, 3))
        p = self.fixed_points

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection = '3d')
        points = np.concatenate((p, x))

        for i in range(len(p)):
            ax.scatter(p[i][0], p[i][1], p[i][2], c='k')
            ax.text(p[i][0], p[i][1], p[i][2], r'$p_{%d}$'%i, size=20, zorder=1, color='k')


        for i in range(len(x)):
            ax.scatter(x[i][0], x[i][1], x[i][2], c='g')
            ax.text(x[i][0], x[i][1], x[i][2], r'$x_{%d}$'%i, size=20, zorder=1, color='k')

        for elem in self.cables:
            a = elem[0]
            b = elem[1]
            cable = np.array([points[a], points[b]]).T
            ax.plot(cable[0], cable[1], cable[2], c='k', linestyle='dashed')

        for elem in self.bars:
            a = elem[0]
            b = elem[1]
            bar = np.array([points[a], points[b]]).T
            ax.plot(bar[0], bar[1], bar[2], c='k')
        return fig, ax


def cable_elastic(state, cables, k):
    l = state[cables[:, 0]]
    r = state[cables[:, 1]]
    diff = np.linalg.norm(l - r, axis=1) - cables[:, 2]
    diff = np.where(diff < 0, 0, diff)
    return (k /(2*cables[:, 2]**2) * diff) @ diff


def bar_elastic(state, bars, c):
    l = state[bars[:, 0]]
    r = state[bars[:, 1]]
    diff = np.linalg.norm(l - r, axis=1) - bars[:, 2]
    return (c /(2*bars[:, 2]**2) * diff) @ diff


def bar_potential(state, bars, rho):
    l = state[bars[:, 0]]
    r = state[bars[:, 1]]
    return 0.5 * rho * bars[:, 2] @ (l[:, 2] + r[:, 2])

def gen_E(cables, bars, free_weights, fixed_points, k, c, rho):
    """Takes in the position of all fixed points p,
    and information about all the cables and bars. Generates the
    objective function used with E_cable_elast

    Returns:
        objective function f(x) where x is a numpy array of free
        optimization variables
    """

    state = np.zeros((len(fixed_points) + len(free_weights), 3))
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

    if len(bars) == 0:
        return func_without_bars

    return func

def gen_grad_E(cables, bars, free_weights, fixed_points, k=2, c=3, rho=0):
    pass


