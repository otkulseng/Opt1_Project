import matplotlib.pyplot as plt
import numpy as np
from algoritmer import testFunksjon
print(testFunksjon())

x = np.linspace(0, 1, 100)
plt.plot(x, -x**2)
plt.savefig("./Bilder/oppg1.png")

dev = 3