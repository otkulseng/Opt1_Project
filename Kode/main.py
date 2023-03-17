import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, approx_fprime
from algoritmer import plot3DTensegrityStructure, gen_E, gen_grad_E

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
print(cables)

bars = np.array([
    # [p[0], x[0], 10],
    # [p[1], x[1], 10],
    # [p[2], x[2], 10],
    # [p[3], x[3], 10]
])

f = gen_E(cables, bars, fw, pp, k=0.1, c=1, rho=0)
grad = gen_grad_E(cables, bars, fw, pp, k=0.1, c=1, rho=0)


x0 = np.ones(3 * len(fw)) * 31
approx = approx_fprime(x0, f, epsilon=1.4901161193847656e-08)

mine = grad(x0)
print(np.matrix(approx))
print(np.matrix(mine))
# res = minimize(f, x0, method='CG', tol=1e-12)
# points = res.x
# sol = np.reshape(points, (-1, 3))
# print(np.round(sol, 6))

# print(sol)

# fig, ax = plot3DTensegrityStructure(sol, pp, cables, bars)
# plt.savefig("./Bilder/oppg1.pdf")
# plt.show()


