from tensegrity import TensegrityStructure
import numpy as np

fixpoints = np.array([
    [5, 5, 0],
    [-5, 5, 0],
    [-5, -5, 0],
    [5, -5, 0]
])

freeweights = np.array([
    1/6,
    1/6,
    1/6,
    1/6
])

# Index lists
p = np.arange(0, len(fixpoints))
x = np.arange(len(fixpoints), len(fixpoints) + len(freeweights))

cables = np.array([
    [p[0], x[0], 3],
    [p[1], x[1], 3],
    [p[2], x[2], 3],
    [p[3], x[3], 3],
    [x[0], x[1], 3],
    [x[1], x[2], 3],
    [x[2], x[3], 3],
    [x[0], x[3], 3]
])

bars = np.array([
])

solution = np.array([
    [2, 2, -3/2],
    [-2, 2, -3/2],
    [-2, -2, -3/2],
    [2, -2, -3/2]
])


# Test for problem 2 - 5
P25 = TensegrityStructure(fixpoints,
                          freeweights,
                          cables,
                          bars,
                          k=3,
                          solution=solution)

fixpoints = np.array([
    [1, 1, 0],
    [-1, 1, 0],
    [-1, -1, 0],
    [1, -1, 0]
])

freeweights = np.array([
    0,
    0,
    0,
    0
])

# Index lists
p = np.arange(0, len(fixpoints))
x = np.arange(len(fixpoints), len(fixpoints) + len(freeweights))

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

# Test for problem 6 - 9
P69 = TensegrityStructure(fixpoints,
                          freeweights,
                          cables,
                          bars,
                          k=0.1,
                          c=1,
                          rho=0)