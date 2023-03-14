import matplotlib.pyplot as plt
import numpy as np
from algoritmer import plot3DTensegrityStructure


# x = np.linspace(0, 1, 100)
# plt.plot(x, -x**2)
# plt.savefig("./Bilder/oppg1.png")
# dev = 3


# Fixed points
pp = np.array([
    [5, 5, 0],
    [-5, 5, 0],
    [-5, -5, 0],
    [5, -5, 0]
])

# Free point weights
fw = np.array([
    1/6,
    1/6,
    1/6,
    1/6
])

# Index lists
p = np.arange(0, len(pp))
x = np.arange(len(pp), len(pp) + len(fw))

cables = np.array([
    [p[0], x[0], 3],
    [p[1], x[1], 3],
    [p[2], x[2], 3],
    [p[3], x[3], 3],
    [x[0], x[1], 3],
    [x[1], x[2], 3],
    [x[2], x[3], 3],
    [x[3], x[0], 3]
])

bars = np.array([
    # [p[0], x[2], 4]
])

# Assume sol is the solution of some optimization algorithm
# sol = BFGS(problem)
sol = np.array([
    [2, 2, -3/2],
    [-2, 2, -3/2],
    [-2, -2, -3/2],
    [2, -2, -3/2]
])

fig, ax = plot3DTensegrityStructure(sol, pp, cables, bars)
plt.savefig("./Bilder/oppg1.png")
plt.show()


