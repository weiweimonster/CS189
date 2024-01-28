import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4963)

X1 = np.random.normal(3, 3, 100)
X2 = 0.5 * X1 + np.random.normal(4, 2, 100)

#Question 1: compute the mean of the sample points
mean1 = np.mean(X1)
mean2 = np.mean(X2)

print("the mean of (X1, X2) are", mean1, mean2)
#the mean of (X1, X2) are 3.314072458757283 5.716452081225764

#Question 2: compute the covariance matrix
sample = np.stack((X1, X2), axis=1)
cov = np.cov(sample.T)
print(cov)
#[[8.74233129 5.20680724]
#[5.20680724 6.8590423 ]]


#Question 3: compute the eigenvalue and eigenvector of the covariance matrix
eigen_value, eigen_vector = np.linalg.eig(cov)
print(eigen_value, eigen_vector)
#Eigenvalue: [13.09195659  2.50941701]
#Eigenvectors:  [[ 0.76745095 -0.64110767], [ 0.64110767  0.76745095]]


#Question 4: Plot 100 points
plt.figure(figsize=(5,5))
plt.scatter(X1, X2)
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.title("Sample points with Eigenvecotrs")
plt.xlabel("X1")
plt.ylabel("X2")
origin = [mean1, mean2]
vector1 = [eigen_vector[0][0]*eigen_value[0], eigen_vector[0][1]*eigen_value[1]]
vector2 = [eigen_vector[1][0]*eigen_value[0], eigen_vector[1][1]*eigen_value[1]]
plt.quiver([mean1, mean1],[mean2, mean2], vector1, vector2, scale=1, angles = 'xy', scale_units= 'xy')
plt.show()

#Question 5: Center and rotate the sample points
sample = sample - np.array([mean1, mean2])
sample = np.dot(eigen_vector.T, sample.T).T
plt.figure(figsize=(5,5))
plt.scatter(sample[:, 0], sample[:, 1])
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.title("Rotated Sample Points")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()