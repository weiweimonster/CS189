import numpy as np
from scipy import io
import scipy.cluster
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd

def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv('submission.csv', index_label='Id')
def train_val_split(train, label, num_val):
    num = train.shape[0]
    num_train = num - num_val
    index = np.random.permutation(num)
    train = train[index]
    label = label[index]
    train_data = train[:num_train]
    train_label = label[:num_train]
    val_data = train[num_train:, :]
    val_label = label[num_train:]
    return train_data, train_label, val_data, val_label

def GDA_train(x, y, mode = "lda"):
    param = {}
    if mode == 'lda':
        labels = np.unique(y)
        cov_matrix = np.zeros((x.shape[1], x.shape[1]))
        for label in labels:
            data = x[(y == label).flatten()]
            num_data = data.shape[0]
            mean = np.mean(data, axis=0)
            cov = np.cov(data.T, bias= False)
            prior = num_data/x.shape[0]
            cov_matrix = np.add(cov_matrix, cov* prior)
            param[label] = [mean, prior]
        for key in param.keys():
            param[key].append(cov_matrix)
    elif mode == 'qda':
        labels = np.unique(y)
        for label in labels:
            data = x[(y == label).flatten()]
            num_data = data.shape[0]
            mean = np.mean(data, axis=0)
            cov = np.cov(data.T, bias= False)
            prior = num_data/x.shape[0]
            param[label] = [mean, prior, cov]
    return param
def make_prediction(param, x, mode = "lda"):
    if mode =='lda':
        pred = []
        for key in param.keys():
            lda = scipy.stats.multivariate_normal(param[key][0], param[key][2],allow_singular=True)
            pdf = lda.logpdf(x)
            posterior = pdf +np.log(param[key][1])
            pred.append(posterior)
        pred = np.array(pred)
    elif mode == "qda":
        pred = []
        for key in param.keys():
            lda = scipy.stats.multivariate_normal(param[key][0], param[key][2],allow_singular=True)
            pdf = lda.logpdf(x)
            posterior = pdf +np.log(param[key][1])
            pred.append(posterior)
    return np.argmax(pred, axis=0)
def evaluate(pred, y):
    y = y.reshape(-1,)
    num_correct = np.sum(pred ==y)
    num = y.shape[0]
    return num_correct/ num

def run_model(mode = 'lda'):
    mnist = io.loadmat("data/mnist_data.mat")
    mnist_x = mnist["training_data"]
    mnist_y = mnist["training_labels"]
    mnist_test = mnist["test_data"]
    mnist_x = scipy.cluster.vq.whiten(mnist_x)
    x, y, val_data, val_label = train_val_split(mnist_x, mnist_y, 10000)
    num_train = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
    error = []
    for num in num_train:
        params = GDA_train(x[:num], y[:num], mode=mode)
        pred = make_prediction(params, val_data, mode=mode)
        accuracy = evaluate(pred, val_label)
        error.append(1-accuracy)
    if mode =='lda':
        plt.plot(np.array(num_train), np.array(error), label="LDA")
        plt.legend()
        plt.xlabel("number of training points")
        plt.ylabel("Error %")
        plt.title("LDA: Number of Training vs Error")
        plt.show()
    else:
        plt.plot(np.array(num_train), np.array(error), label="QDA")
        plt.legend()
        plt.xlabel("number of training points")
        plt.ylabel("Error %")
        plt.title("QDA: Number of Training vs Error")
        plt.show()

def error_of_each_class(mode = 'lda'):
    mnist = io.loadmat("data/mnist_data.mat")
    mnist_x = mnist["training_data"]
    mnist_y = mnist["training_labels"]
    mnist_test = mnist["test_data"]
    mnist_x = scipy.cluster.vq.whiten(mnist_x)
    x, y, val_data, val_label = train_val_split(mnist_x, mnist_y, 10000)
    num_train = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
    error = {}
    for num in num_train:
        params = GDA_train(x[:num], y[:num], mode=mode)
        pred = make_prediction(params, val_data, mode=mode)
        for label in np.unique(val_label):
            index = (val_label==label).flatten()
            accuracy = evaluate(pred[index], val_label[index])
            if label in error:
                error[label].append(1-accuracy)
            else:
                error[label] = [1-accuracy]
    if mode == 'lda':
        for key in error.keys():
            plt.plot(np.array(num_train), np.array(error[key]), label=key)
        plt.legend()
        plt.xlabel("number of training points")
        plt.ylabel("Error %")
        plt.title("LDA Classification Digitwise")
        plt.show()
    else:
        for key in error.keys():
            plt.plot(np.array(num_train), np.array(error[key]), label=key)
        plt.legend()
        plt.xlabel("number of training points")
        plt.ylabel("Error %")
        plt.title("QDA Classification Digitwise")
        plt.show()
def mnist_test_prediction(num_train = 30000, mode = 'lda'):
    mnist = io.loadmat("data/mnist_data.mat")
    mnist_x = mnist["training_data"]
    mnist_y = mnist["training_labels"]
    mnist_test = mnist["test_data"]
    mnist_x = scipy.cluster.vq.whiten(mnist_x)
    mnist_test = scipy.cluster.vq.whiten(mnist_test)
    index = np.random.permutation(mnist_x.shape[0])
    mnist_x = mnist_x[index]
    mnist_y = mnist_y[index]
    params = GDA_train(mnist_x, mnist_y, mode=mode)
    pred = make_prediction(params, mnist_test, mode=mode)
    results_to_csv(pred)
def spam_test_prediction(num_train = 5000, mode = 'lda'):
    spam = io.loadmat("data/spam_data.mat")
    spam_x = spam["training_data"]
    spam_y = spam["training_labels"]
    spam_test = spam["test_data"]
    spam_x = scipy.cluster.vq.whiten(spam_x)
    spam_test = scipy.cluster.vq.whiten(spam_test)
    index = np.random.permutation(spam_x.shape[0])
    spam_x = spam_x[index]
    spam_y = spam_y[index]
    params = GDA_train(spam_x, spam_y, mode=mode)
    pred = make_prediction(params, spam_test, mode=mode)
    results_to_csv(pred)
# run_model("qda")
# error_of_each_class('lda')
# mnist_test_prediction()
# spam_test_prediction(mode='qda')