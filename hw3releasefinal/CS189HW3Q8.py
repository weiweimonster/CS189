import numpy as np
from scipy import io
import scipy
import scipy.cluster
import matplotlib.pyplot as plt


mnist = io.loadmat("data/mnist_data.mat")
mnist_x = mnist["training_data"]
mnist_y = mnist["training_labels"]
mnist_test = mnist["test_data"]
mnist_x = scipy.cluster.vq.whiten(mnist_x)
mean = np.mean(mnist_x[(mnist_y == 3).flatten()], axis = 0)
cov = np.cov(mnist_x.T)
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(cov)
f.add_subplot(1, 2, 2)
plt.imshow(mean.reshape(28, 28))
plt.show()
