import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from algoritmer import plot3DTensegrityStructure, gen_E


# x = np.linspace(0, 1, 100)
# plt.plot(x, -x**2)
# plt.savefig("./Bilder/oppg1.png")
# dev = 3


# Fixed points
pp = np.array([
    [1, 1, 0],
    [-1, 1, 0],
    [-1, -1, 0],
    [1, -1, 0]
])

# Free point weights
fw = np.array([
    0,
    0,
    0,
    0
])

# Index lists
p = np.arange(0, len(pp))
x = np.arange(len(pp), len(pp) + len(fw))

cables = np.array([
    [p[0], x[3], 8],
    [p[1], x[0], 8],
    [p[2], x[1], 8],
    [p[3], x[2], 8],
    [x[0], x[3], 1],
    [x[0], x[1], 1],
    [x[1], x[2], 1],
    [x[2], x[3], 1]
])

bars = np.array([
    [p[0], x[0], 10],
    [p[1], x[1], 10],
    [p[2], x[2], 10],
    [p[3], x[3], 10]
])

f = gen_E(pp, cables, fw, bars, k=0.1, c=1, rho=0)


x0 = np.ones(3 * len(fw))
res = minimize(f, x0, method='CG')
points = res.x
sol = np.reshape(points, (-1, 3))
print(sol)

# print(sol)

# Assume sol is the solution of some optimization algorithm
# sol = BFGS(problem)
# sol = np.array([
#     [2, 2, -3/2],
#     [-2, 2, -3/2],
#     [-2, -2, -3/2],
#     [2, -2, -3/2]
# ])


fig, ax = plot3DTensegrityStructure(sol, pp, cables, bars)
plt.savefig("./Bilder/oppg1.png")
plt.show()


