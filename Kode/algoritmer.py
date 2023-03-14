from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

def plot3DTensegrityStructure(x, p, cables, bars):
    """Generates a 3D image of the tensegrity structure. Usage:
    // fixed points
    p = np.array([
        [1, 1, 2],
        [2, 2, 1],
        [1, -1, 4]
    ])

    // free points
    x = np.array([
        [0, 0, 0],
        [0, 1, -1],
    ])

    // cables. First element means cable from p0 to x1
    cables = np.array([
        [p[0], x[1]],
        [p[1], x[0]],
        [p[1], x[1]],
        [x[1], x[0]]
    ])

    // bars. First element means bar from p0 to x0
    bars = np.array([
        [p[0], x[0]]
    ])

    fig, ax = plot3DTensegrityStructure(x, p, cables, bars)
    plt.show()
    """

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection = '3d')

    pp = np.transpose(p)
    ax.scatter(pp[0], pp[1], pp[2], c='k')

    xx = np.transpose(x)
    ax.scatter(xx[0], xx[1], xx[2], c='g')

    for cable in cables:
        cable = np.transpose(cable)
        ax.plot(cable[0], cable[1], cable[2], c='k', linestyle='dashed')

    for bar in bars:
        bar = np.transpose(bar)
        ax.plot(bar[0], bar[1], bar[2], c='k')
    return fig, ax

