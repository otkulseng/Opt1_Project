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
    [2, 2, 0],
    [0, 2, 0]
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
    [p[3], x[0], 2]
])

LOCALMIN = TensegrityStructure(fixpoints,
                          freeweights,
                          cables,
                          bars,
                          c=1)


fixpoints = np.array([
    # [1, 1, 0],
])

freeweights = np.zeros(8, dtype=np.float64)

for i in range(4):
    freeweights[i] = 1e-4

# Index lists
cables = np.array([
    [0, 7, 8],
    [1, 4, 8],
    [2, 5, 8],
    [3, 6, 8],
    [4, 5, 1],
    [5, 6, 1],
    [6, 7, 1],
    [4, 7, 1],
    [0, 1, 2],
    [1, 2, 2],
    [2, 3, 2],
    [0, 3, 2]
])


bars = np.array([
    [0, 4, 10],
    [1, 5, 10],
    [2, 6, 10],
    [3, 7, 10]
])


FREESTANDING = TensegrityStructure(fixpoints,
                          freeweights,
                          cables,
                          bars,
                          k=0.1,
                          c=1,
                          rho=1e-10,
                          quadratic_penalization=True)

def STORIES(N):
    fixpoints = np.array([
    ])
    freeweights = np.zeros(4 + 4 * N, dtype=np.float64)

    for i in range(4):
        freeweights[i] = 1e-4

    cables = []
    for i in range(N+1):
        cables.append([0 + 4*i, 1 + 4*i, 1])
        cables.append([1 + 4*i, 2 + 4*i, 1])
        cables.append([2 + 4*i, 3 + 4*i, 1])
        cables.append([0 + 4*i, 3 + 4*i, 1])
        if i < N:
            cables.append([0 + 4*i, 7 + 4*i, 8])
            cables.append([1 + 4*i, 4 + 4*i, 8])
            cables.append([2 + 4*i, 5 + 4*i, 8])
            cables.append([3 + 4*i, 6 + 4*i, 8])
    cables = np.array(cables)

    bars = []
    for i in range(len(freeweights)):
        if i + 4 == len(freeweights):
            break
        bars.append([i, i+4, 10])
    bars = np.array(bars)


    return TensegrityStructure(fixpoints,
                            freeweights,
                            cables,
                            bars,
                            k=0.1,
                            c=1,
                            rho=1e-10,
                            quadratic_penalization=True)



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
                          quadratic_penalization=True)


l = 30
h = 40
m = int(2*h/3)
u = int(np.sqrt(0.5*l**2 + m**2))

fixpoints = np.array([
])
freeweights = np.ones(10)
for i in range(4):
    freeweights[i] = 1/10

cables = np.array([
    [8, 9, 2],
    [0, 4, h],
    [1, 5, h],
    [2, 6, h],
    [3, 7, h]
])

bars = np.array([
    [0, 1, l],
    [1, 2, l],
    [2, 3, l],
    [0, 3, l],
    [4, 5, l],
    [5, 6, l],
    [6, 7, l],
    [4, 7, l],
    [1, 9, u],
    [7, 8, u]
])

TABLE = TensegrityStructure(fixpoints,
                            freeweights,
                            cables,
                            bars,
                            k=1e10,
                            c=1e5,
                            # rho=freeweights[0]/30,
                            quadratic_penalization=True)

