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


def helper_split(p, cables):
    cable_len = cables[:, -1]
    cables = np.array(cables[:, :2], dtype=int)

    # Cables between free points
    free_free = cables[(cables[:, 1] >= len(p)) & (cables[:, 0] >= len(p))]
    free_free_len = cable_len[(cables[:, 1] >= len(p)) & (cables[:, 0] >= len(p))]
    free_free -= len(p) # Index in x¨

    # Cables between free and fixed point
    fix_free = cables[(cables[:, 1] >= len(p)) & (cables[:, 0] < len(p))]
    fix_free_len = cable_len[(cables[:, 1] >= len(p)) & (cables[:, 0] < len(p))]
    fix_free[:, 1] -= len(p)
    return free_free, free_free_len, fix_free, fix_free_len


def gen_E(p, cables, free_weights, bars, k=2, c=3, rho=0):
    """Takes in the position of all fixed points p,
    and information about all the cables and bars. Generates the
    objective function used with E_cable_elast

    Returns:
        objective function f(x) where x is a numpy array of free
        optimization variables
    """

    ff_cables, ff_cables_len, fx_cables, fx_cables_len = helper_split(p, cables)
    ff_bars, ff_bars_len, fx_bars, fx_bars_len = helper_split(p, bars)


    def func(xx):
        x = np.reshape(xx, (-1, 3))

        # Energy of cables between fixed and free point
        pfree = p[fx_cables[:, 0]]
        xfree = x[fx_cables[:, 1]]
        a = np.linalg.norm(pfree-xfree, axis=1) - fx_cables_len
        a = np.where(a<0, 0, a)

        Efixfree = (k /(2*fx_cables_len**2) * a) @ a

        # Energy of cables between free and free point
        xfree1 = x[ff_cables[:, 0]]
        xfree2 = x[ff_cables[:, 1]]
        b = np.linalg.norm(xfree1-xfree2, axis=1) - ff_cables_len
        b = np.where(b<0, 0, b)

        Efreefree = (k /(2*ff_cables_len**2) * b) @ b

        # Energy of all free weights
        Eext = free_weights @ x[:, -1]

        # Energy of bars between fixed and free point
        pfree1 = p[fx_bars[:, 0]]
        xfree1 = x[fx_bars[:, 1]]
        aa = np.linalg.norm(pfree1-xfree1, axis=1) - fx_bars_len

        Ebar_fix_free = (c /(2*fx_bars_len**2) * aa) @ aa

        # Energy of cables between free and free point
        xfree11 = x[ff_bars[:, 0]]
        xfree22 = x[ff_bars[:, 1]]
        bb = np.linalg.norm(xfree11-xfree22, axis=1) - ff_bars_len

        Ebar_free_free = (c /(2*ff_bars_len**2) * bb) @ bb

        return Efixfree + Efreefree + Eext + Ebar_fix_free + Ebar_free_free
    return func



