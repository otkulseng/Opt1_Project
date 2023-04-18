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



x0 = +np.array([1, 1, 1])

ts = TEST.LOCALMIN
res = bfgs(x0, ts.func, ts.grad, Niter=100, plot_summary=True)

ts.plot(res)
plt.show()

# 0.15280676109681207
# -0.17294678450908882



