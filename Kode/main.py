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




ts = TEST.FREESTANDING
x0 = np.arange(3 * len(ts.free_weights))

# xx = x0
# x = np.reshape(xx, (-1, 3))
# grad = np.zeros(x.shape).T
# z = x[:, 2].copy()
# print(z)
# grad[-1] = np.where(z < 0, -z, 0)
# print(grad.T)
# print(x)

                # return orig(x) + mu * (grad.T).flatten()

res, conv = bfgs(x0, ts.func, ts.grad, Niter=500, quadratic_penalty=True, plot_summary=True, convergence_plot=True)

plt.plot(conv)
ts.plot(res)
plt.show()







