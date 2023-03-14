from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

def plot3DTensegrityStructure(x, p, cables, bars):
    """Generates a 3D image of the tensegrity structure. Usage:
    # Fixed points
    pp = np.array([
        [1, 1, 2],
        [2, 2, 1],
        [1, -1, 4]
    ])

    # Free points
    num_free_points = 2

    # Index lists
    p = np.arange(0, len(pp))
    x = np.arange(len(pp), len(pp) + num_free_points)

    # First element means cable from p0 to x1 with rest length 3
    cables = np.array([
        (p[0], x[1], 3),
        (p[1], x[0], 2),
        (p[1], x[1], 4),
        (x[1], x[0], 1)
    ])

    bars = np.array([
        (p[0], x[0], 4)
    ])

    # Assume xx is the solution of some optimization algorithm
    # xx = BFGS(problem)
    xx = np.array([
        [0, 0, 0],
        [0, 1, -1],
    ])

    fig, ax = plot3DTensegrityStructure(xx, pp, cables, bars)
    plt.show()
    """

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection = '3d')

    points = np.concatenate((p, x))

    for i in range(len(p)):
        ax.scatter(p[i][0], p[i][1], p[i][2], c='k')
        ax.text(p[i][0], p[i][1], p[i][2], r'$p_{%d}$'%i, size=20, zorder=1, color='k')


    for i in range(len(x)):
        ax.scatter(x[i][0], x[i][1], x[i][2], c='g')
        ax.text(x[i][0], x[i][1], x[i][2], r'$x_{%d}$'%i, size=20, zorder=1, color='k')

    for elem in cables:
        a = elem[0]
        b = elem[1]
        cable = np.array([points[a], points[b]]).T
        ax.plot(cable[0], cable[1], cable[2], c='k', linestyle='dashed')

    for elem in bars:
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

def gen_E(p, cables, free_weights, bars, k=2, c=3, rho=0):
    """Takes in the position of all fixed points p,
    and information about all the cables and bars. Generates the
    objective function used with E_cable_elast

    Returns:
        objective function f(x) where x is a numpy array of free
        optimization variables
    """
    state = np.zeros((len(p) + len(free_weights), 3))
    state[:len(p)] = p

    def func(xx):
        x = np.reshape(xx, (-1, 3))
        state[len(p):] = x

        # Elastic energy of the cables
        E_cable_elastic = cable_elastic(state, cables, k)

        # Elastic energy of the bars
        E_bar_elastic = bar_elastic(state, bars, c)

        # Potential energy of the bars
        E_bar_potential = bar_potential(state, bars, rho)

        # Potential energy of all free weights
        E_ext = free_weights @ x[:, 2]

        return E_cable_elastic + E_bar_elastic + E_bar_potential + E_ext
    return func



