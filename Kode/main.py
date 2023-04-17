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

# res = bfgs(x0, ts.func, Niter=150)
# res = np.reshape(res,(-1, 3))
# print(res)

ts = TEST.P69
x0 = np.ones(3 * len(ts.free_weights))
res = minimize(ts.func, x0, tol=1e-20, method="BFGS")
res = np.reshape(res.x, (-1, 3))
print(np.round(res, 6))


# fig, ax = ts.plot(res)
# plt.savefig("./Bilder/oppg1.pdf")
# plt.show()





