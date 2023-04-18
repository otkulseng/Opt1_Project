import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, approx_fprime
import tests as TEST
from algoritmer import bfgs


# x0 = np.array([
#     [-7.09729653e-01,-2.19841562e-05,9.54286612e+00],
#  [-1.92495992e-05 ,-7.09732144e-01 , 9.54286984e+00],
#  [ 7.09690896e-01, -2.17653342e-05,  9.54287599e+00],
#  [-1.95179315e-05  ,7.09688419e-01 , 9.54287231e+00]], dtype=np.float64)
# x0 = x0.flatten()
# print(res)






ts = TEST.SANITYCHECK
x0 = np.arange(3 * len(ts.free_weights))
muMax = 10
for mu in np.linspace(1, muMax, 20):
    # x0 = bfgs(x0, ts.func(mu), ts.grad(mu), Niter=100)
    x0 = minimize(ts.func(mu), x0, jac=ts.grad(mu), method="CG", tol=1e-12).x

res = bfgs(x0, ts.func(muMax), ts.grad(muMax), Niter=0, plot_summary=True)
# res = x0


ts.plot(res)

res = np.array([
    [10, 10, 0],
    [10, -10, 0],
    [-10, -10, 0],
    [-10, 10, 0]
])

# ts.plot(res)

norm = np.linalg.norm(ts.grad(muMax)(res))
print(norm)

norm = np.linalg.norm(ts.func(muMax)(res))
print(norm)

plt.show()







