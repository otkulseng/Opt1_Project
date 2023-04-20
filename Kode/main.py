import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, approx_fprime
import tests as TEST
from algoritmer import bfgs



ts = TEST.FREESTANDING2
x0 = np.arange(3 * len(ts.free_weights))

mu = 1
res = x0.copy()

prev = x0
for i in range(10):
    res, num = bfgs(res, ts.func(mu), ts.grad(mu), Niter=1000, return_iteration=True)
    print(f' {i}: {num}, {mu} ')

    mu *= 1.5
    if num < 1000:
        mu *= 2
    if num < 500:
        mu *= 2
    if num < 250:
        mu *= 2

    mu = min(mu, 1e10)

    if np.linalg.norm(res - prev) < 1e-12:
        break
    prev = res.copy()


res = bfgs(res, ts.func(mu), ts.grad(mu), Niter=1000, plot_summary=True)
ts.plot(res)
plt.show()
