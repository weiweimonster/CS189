import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
from scipy.special import expit
import pandas as pd

np.random.seed(1130) #set the random seed to 1130


def sigmoid( x):
    return 1 / (1 + np.exp(-x))
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv('submission.csv', index_label='Id')
class Q2_4():
    def __init__(self):
        self.x1 = np.array([0.2,3.1, 1])
        self.x2 = np.array([1.0,3.0, 1])
        self.x3 = np.array([-0.2,1.2, 1])
        self.x4 = np.array([1.0,1.1, 1])
        self.X = np.stack((self.x1, self.x2, self.x3, self.x4))
        self.w0 = np.array([-1, 1, 0])
        self.w = [self.w0]
        self.y = np.array([1,1,0,0])
    def calculate_s(self):
        curr = len(self.w) - 1
        curr_w = self.w[curr]
        return sigmoid(np.dot(self.X, curr_w))
    def update_w(self):
        curr = len(self.w) - 1
        curr_w = self.w[curr]
        s_i = self.calculate_s()
        intervalue = np.dot(self.X.T, np.diag(np.multiply(s_i, (1-s_i))))
        hessian =np.dot(intervalue, self.X)
        next_w = curr_w + np.dot(np.dot(np.linalg.inv(hessian), self.X.T), (self.y - s_i))
        self.w.append(next_w)
        return next_w
class gradient_descent():
    def __init__(self, alpha, l):
        wine_data = io.loadmat("data_wine.mat")
        X = whiten(wine_data['X']- np.mean(wine_data['X'], axis = 0))
        test = whiten(wine_data['X_test']- np.mean(wine_data['X_test'], axis = 0))
        self.alpha = alpha
        self.l = l
        self.label = wine_data['y']
        self.test = np.concatenate((test, np.ones((test.shape[0], 1))), axis=1)
        self.num_train, num_features = X.shape
        self.X = np.concatenate((X, np.ones((self.num_train, 1))), axis=1)
        self.w = np.zeros((num_features +1,))
        self.costs = []
    def validation_split(self, num = 1200):
        index = np.random.permutation(self.num_train)
        self.X = self.X[index]
        self.label = self.label[index]
        self.val_data = self.X[:num]
        self.val_label = self.label[:num]
        self.X = self.X[num:]
        self.label = self.label[num:]
    def compute_cost_append(self):
        X= np.dot(self.X, self.w)
        s = expit(X)
        cost = -np.dot(self.label.T, np.log(s)) - np.dot((1 - self.label).T, np.log(1-s)) + self.l/2 * np.linalg.norm(self.w)
        self.costs.append(cost)
    def make_prediction(self, val=False):
        if val:
            pred = np.where(np.dot(self.val_data, self.w) >= 0.5, 1, 0)
        else:
            pred = np.where(np.dot(self.test, self.w) >= 0.5, 1, 0)
        return pred
    def accuracy(self, pred, y):
        y = y.reshape(-1,)
        correct = np.sum(pred == y)
        num = y.shape[0]
        return correct/num
    def Batch_gradient_descent(self, num_iteration = 5000, submit=False):
        alpha = self.alpha
        l = self.l
        self.validation_split()
        self.compute_cost_append()
        for i in range(num_iteration):
            s = expit(np.dot(self.X, self.w))
            diff = l*self.w - np.dot(self.X.T, self.label.reshape(-1, ) - s)
            self.w -= alpha * diff
            self.compute_cost_append()
        if submit:
            pred = self.make_prediction(False)
            results_to_csv(pred)
        acc = self.accuracy(pred, self.val_label)
        print(acc)
        plt.plot(np.arange(num_iteration+1), self.costs)
        plt.title("Training error vs iterations")
        plt.xlabel("Number of iteration")
        plt.ylabel("Cost")
        plt.show()
    def Stochastic_Gradient_Descent(self,num_iteration=5000, decay=False):
        alpha = self.alpha
        l = self.l
        self.validation_split()
        self.compute_cost_append()
        for i in range(num_iteration):
            idx = i % self.X.shape[0]
            s = expit(np.dot(self.X[idx, :], self.w))
            s = self.label[idx] - s
            sample =  self.X[idx],
            diff = l * self.w - self.X.shape[0] * np.dot(s, sample)
            if decay:
                adj_alpha = 20 * alpha / num_iteration
            else:
                adj_alpha = alpha
            self.w -= adj_alpha * diff
            self.compute_cost_append()
        pred = self.make_prediction()
        acc = self.accuracy(pred, self.val_label)
        print(acc)
        plt.plot(np.arange(num_iteration + 1), self.costs)
        plt.title("Training error vs iterations")
        plt.xlabel("Number of iteration")
        plt.ylabel("Cost")
        plt.show()
class isocontours():
    def __init__(self, p):
        self.grid = np.mgrid[-5:5:200j, -5:5:200j]
        self.p = p
    def p_norm(self, ):
        p = self.p
        x = self.grid[0]
        y = self.grid[1]
        x_norm = np.float_power(np.abs(x), p)
        y_norm = np.float_power(np.abs(y), p)
        total_norm = x_norm + y_norm
        return np.float_power( total_norm, 1/p)
    def plot_contours(self):
        norm = self.p_norm()
        plt.contour(self.grid[0],self.grid[1], norm)
        plt.title("Isocontours of l-"+ str(self.p) + " norm" )
        plt.show()




def main():
    ############### Question 2.4 #################################
    # q2 = Q2_4()
    # s0 = q2.calculate_s()
    # w1 = q2.update_w()
    # s1 = q2.calculate_s()
    # w2 = q2.update_w()
    # print(s0)
    # print(w1)
    # print(s1)
    # print(w2)
    ############### Question 3.2 #################################
    # BGD = gradient_descent(0.001,0.1)
    # BGD.Batch_gradient_descent()

    ############### Question 3.4 #################################
    # SGD = gradient_descent(0.000001, 0.1)
    # SGD.Stochastic_Gradient_Descent()

    ############### Question 3.5 #################################
    # SGD_adj = gradient_descent(0.001, 0.1)
    # SGD_adj.Stochastic_Gradient_Descent(num_iteration= 5000, decay=True)

    ############### Question 3.6 #################################
    # BGD = gradient_descent(0.001,0.1)
    # BGD.Batch_gradient_descent(submit=True)

    ############### Question 3.6 #################################
    # ps = [0.5, 1, 2]
    # ls = [isocontours(p) for p in ps]
    # for l in ls:
    #     l.plot_contours()
main()