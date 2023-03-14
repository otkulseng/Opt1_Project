import matplotlib.pyplot as plt
import numpy as np
from algoritmer import plot3DTensegrityStructure


# x = np.linspace(0, 1, 100)
# plt.plot(x, -x**2)
# plt.savefig("./Bilder/oppg1.png")
# dev = 3


p = np.array([
    [1, 1, 2],
     [2, 2, 1],
     [1, -1, 4]
     ])

x = np.array([
    [0, 0, 0],
    [0, 1, -1],
])

cables = np.array([
    [p[0], x[1]],
    [p[1], x[0]],
    [p[1], x[1]],
    [x[1], x[0]]
])

bars = np.array([
    [p[0], x[0]]
])




fig, ax = plot3DTensegrityStructure(x, p, cables, bars)
plt.show()

