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
], dtype=np.float64)

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
                          rho=0.0)



fixpoints = np.array([
    [0, 0, 0],
    [2, 0, 0],
    [1, np.sqrt(3), 0],
])

freeweights = np.array([
    0.1
], dtype=np.float64)

# Index lists
p = np.arange(0, len(fixpoints))
x = np.arange(len(fixpoints), len(fixpoints) + len(freeweights))

cables = np.array([
])

bars = np.array([
    [p[0], x[0], 2],
    [p[1], x[0], 2],
    [p[2], x[0], 2],
])

LOCALMIN = TensegrityStructure(fixpoints,
                          freeweights,
                          cables,
                          bars,
                          c=1)


fixpoints = np.array([
])

freeweights = np.zeros(8)

freeweights[0] = 10
freeweights[1] = 10
freeweights[2] = 10
freeweights[3] = 10


cables = np.array([
    [1, 2, 4],
    [2, 3, 4],
    [3, 4, 4],
    [1, 4, 4],
    [1, 8, 9],
    [2, 5, 9],
    [3, 6, 9],
    [4, 7, 9],
    [5, 6, 2],
    [6, 7, 2],
    [7, 8, 2],
    [5, 8, 2]
])-1


bars = np.array([
    [1, 5, 11],
    [2, 6, 11],
    [3, 7, 11],
    [4, 8, 11]
])-1


FREESTANDING = TensegrityStructure(fixpoints,
                          freeweights,
                          cables,
                          bars,
                          k=0.1,
                          c=1,
                          rho=1e-3,
                          mu=1)



fixpoints = np.array([
    [10, 10, 20.1],
    [10, -10, 20.1],
    [-10, -10, 20.1],
    [-10, 10, 20.1]
])

freeweights = np.ones(4)/100



cables = np.array([
    [0, 4, 20],
    [1, 5, 20],
    [2, 6, 20],
    [3, 7, 20],
    [4, 5, 20],
    [5, 6, 20],
    [6, 7, 20],
    [4, 7, 20]
])

bars = np.array([
])


SANITYCHECK = TensegrityStructure(fixpoints,
                          freeweights,
                          cables,
                          bars,
                          k=0.1,
                          c=1,
                          mu=1)

