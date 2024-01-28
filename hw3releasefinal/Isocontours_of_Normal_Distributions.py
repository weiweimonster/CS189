import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt

# 3
def plot_contour(x, y, mean, variance):
    pos = np.dstack((x, y))
    rv = stat.multivariate_normal(mean, variance)
    z = rv.pdf(pos)
    plt.contourf(x, y, z)
    plt.colorbar()
    plt.show()
def plot_subtract_contour(x, y, mean1, mean2, variance1, variance2):
    pos = np.dstack((x, y))
    rv1 = stat.multivariate_normal(mean1, variance1)
    rv2 = stat.multivariate_normal(mean2, variance2)
    z = rv1.pdf(pos) - rv2.pdf(pos)
    plt.contourf(x, y, z)
    plt.colorbar()
    plt.show()





#3.1
# x1, y1 = np.mgrid[-2:4:0.01, -2:4:0.01]
# mean1, variance1 = [1,1], [[1,0],[0,2]]
# plot_contour(x1, y1, mean1, variance1)

#3.2
# x2, y2 = np.mgrid[-4:2:0.01, -3:6:0.01]
# mean2, variance2 = [-1,2], [[2,1],[1,4]]
# plot_contour(x2, y2, mean2, variance2)

#3.3
# x3, y3 = np.mgrid[-4:6:0.01, -3:6:0.01]
# mean3_1, mean3_2, variance3 = [0,2], [2,0],  [[2,1],[1,1]]
# plot_subtract_contour(x3, y3, mean3_1,mean3_2, variance3, variance3)

#3.4
# x4, y4 = np.mgrid[-4:6:0.01, -3:6:0.01]
# mean4_1, mean4_2, variance4_1, variance4_2 = [0,2], [2,0],  [[2,1],[1,1]], [[2,1],[1,4]]
# plot_subtract_contour(x4, y4, mean4_1, mean4_2, variance4_1, variance4_2)

#3.5
# x5, y5 = np.mgrid[-6:6:0.01, -6:6:0.01]
# mean5_1, mean5_2, variance5_1, variance5_2 = [1,1], [-1,-1],  [[2,0],[0,1]], [[2,1],[1,2]]
# plot_subtract_contour(x5, y5, mean5_1, mean5_2, variance5_1, variance5_2)