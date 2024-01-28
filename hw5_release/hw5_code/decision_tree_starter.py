from collections import Counter
import csv
import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate
import sklearn.tree
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import random
import matplotlib.pyplot as plt

random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number


# Vectorized function for hashing for np efficiency
def w(x):
    return np.int(hash(x)) % 1000


h = np.vectorize(w)


def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv('submission.csv', index_label='Id')


class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None, m=False):
        # TODO implement __init__ function
        self.max_depth = max_depth
        self.features = feature_labels
        ############# Onle NON-Leaf node has it######################
        self.split_thresh = None
        self.split_feature_idx = None
        self.left = self.right = None
        ############# Only leaf node has below#####################
        self.data = None
        self.label = None
        self.pred = None

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO implement information gain function
        idx = X < thresh
        left, right = y[idx], y[~idx]
        return -len(left) * DecisionTree.entropy(X, left, thresh) - len(right) * DecisionTree.entropy(X, right, thresh)

    @staticmethod
    def entropy(X, y, thresh):
        # TODO implement entropy (or optionally gini impurity) function
        cardinality = len(y)
        labels, counts = np.unique(y, return_counts=True)
        counts = counts.astype(float) / cardinality
        counts = counts * np.log(counts)
        return -np.sum(counts)

    def split(self, X, y, idx, thresh):
        # TODO implement split function
        index = X[:, idx] < thresh
        return X[index, :], X[~index, :], y[index], y[~index]

    def fit(self, X, y, Randomf=False, p=None):
        # TODO implement fit function
        if self.max_depth > 0:
            num_features = len(self.features)
            thresholds = []
            max_info_gain = -float("inf")
            if not Randomf:  ##normal decision tree
                for i in range(num_features):
                    thresholds.append(np.unique(X[:, i]) + eps)
                    for thresh in thresholds[i]:
                        info_gain = DecisionTree.information_gain(X[:, i], y, thresh)
                        if info_gain > max_info_gain:
                            max_info_gain = info_gain
                            self.split_feature_idx = i
                            self.split_thresh = thresh
            else:  ##random forest
                sample_feature_idx = random.sample(list(range(X.shape[1])), p)
                count = 0
                for idx in sample_feature_idx:
                    thresholds.append(np.unique(X[:, idx]) + eps)
                    for thresh in thresholds[count]:
                        info_gain = DecisionTree.information_gain(X[:, idx], y, thresh)
                        if info_gain > max_info_gain:
                            max_info_gain = info_gain
                            self.split_feature_idx = idx
                            self.split_thresh = thresh
                    count += 1

            X_left, X_right, y_left, y_right = self.split(X, y, self.split_feature_idx, self.split_thresh)
            if X_left.shape[0] > 0 and X_right.shape[0] > 0:
                self.left = DecisionTree(self.max_depth - 1, self.features)
                # self.left.num0 = 1 - np.sum(y_left)/y_left.size
                # self.left.num1 = np.sum(y.left)/y_left.size
                self.left.fit(X_left, y_left, p=p, Randomf=Randomf)
                self.right = DecisionTree(self.max_depth - 1, self.features)
                # self.right.num0 = 1 - np.sum(y_right)/y_right.size
                # self.right.num1 = np.sum(y_right)*y_right.size
                self.right.fit(X_right, y_right, p=p, Randomf=Randomf)
            else:
                self.max_depth = 0
                self.data, self.label = X, y
                self.pred = Counter(y).most_common(1)[0][0]
        else:
            self.max_depth = 0
            self.data, self.label = X, y
            self.pred = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, X, verbose=False):
        # TODO implement predict function
        if len(X.shape) != 2:
            X = X.reshape(-1, X.shape[0])
        if self.max_depth == 0:
            if verbose:
                print("This is a leaf node and the prediction is {pred}".format(pred=self.pred))
            return self.pred * np.ones(X.shape[0])
        else:
            pred = np.ones(X.shape[0])
            idx = X[:, self.split_feature_idx] < self.split_thresh
            if verbose:
                print(
                    "The feature that is being split on this node is {feature_name} and the threshold is {thresh}".format(
                        feature_name=self.features[self.split_feature_idx], thresh=self.split_thresh))
            if X[idx, :].shape[0] != 0:
                pred[idx] = self.left.predict(X[idx, :], verbose=verbose)
            if X[~idx, :].shape[0] != 0:
                pred[~idx] = self.right.predict(X[~idx, :], verbose=verbose)
            return pred


class RandomForest(DecisionTree):
    def __init__(self, n=200, m=1, num_sample=5000, p=20, max_depth=20, feature_labels=None):
        '''  n (int) is the number of trees in the random forest
            num_sample (int) is the number of samples drawn with replacement from the original data
            p (int) is the number of feature to be randomly sample WITHOUT replacement from the feature_labels
            max_depth (int) is the maximum depth of the tree
        '''
        # TODO implement function
        super().__init__(max_depth=max_depth, feature_labels=feature_labels)
        self.n = n;
        self.forest = [DecisionTree(max_depth=self.max_depth, feature_labels=feature_labels) for i in range(self.n)]
        self.num_sample = num_sample
        self.p = p

    def fit(self, X, y, Randomf=True, p=None):
        for i in range(self.n):
            bagg_idx = random.choices(np.arange(X.shape[0]), k=self.num_sample)
            X_train, y_train = X[bagg_idx, :], y[bagg_idx]
            self.forest[i].fit(X_train, y_train, Randomf=Randomf, p=self.p)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.forest]).T
        pred = []
        for i in range(predictions.shape[0]):
            pred.append(Counter(predictions[i]).most_common(1)[0][0])
        return pred


def cross_validation(X, y, cv=5, tree=DecisionTree, p=None, n=20, max_depth=30, dataset="spam", features=None):
    if dataset == "titanic":
        features = X[0, :]
        X = X[1:, :]
    shuffle_index = np.random.permutation(X.shape[0])
    X = X[shuffle_index]
    y = y.reshape(X.shape[0], -1)[shuffle_index]
    Xy = np.concatenate((X, y), axis=1)
    cv_set = np.array_split(Xy, cv)
    train_acc = []
    val_acc = []
    for i in range(cv):
        X_val, y_val = cv_set[i][:, :-1], cv_set[i][:, -1]
        Xy_train = cv_set[:i] + cv_set[i + 1:]
        Xy_train = np.vstack(Xy_train)
        X_train, y_train = Xy_train[:, :-1], Xy_train[:, -1]
        if tree == RandomForest:
            dt = RandomForest(max_depth=max_depth, feature_labels=features, p=p, n=n)
        else:
            dt = tree(max_depth=max_depth, feature_labels=features)
        dt.fit(X_train, y_train)
        pred = dt.predict(X_train)
        y_train = np.array(y_train, dtype=np.float)
        y_val = y_val.astype("float")
        train_acc.append(sklearn.metrics.accuracy_score(y_train, pred))
        val_acc.append(sklearn.metrics.accuracy_score(y_val, dt.predict(X_val)))
    print("The average training accuracy is {avg_train}".format(avg_train=sum(train_acc) / len(train_acc)))
    print("The average validation accuracy is {avg_val}".format(avg_val=sum(val_acc) / len(val_acc)))
    return sum(train_acc) / len(train_acc), sum(val_acc) / len(val_acc)


def prepross(data, key_wanted, test_data=None, one_hot=True, train_size=None, test=False):
    ''' data(numpy array): the data that needs to be processed
        key(tuple of keys that you wish to one hot encoding)'''
    # Infer missing value
    data = data.reshape(data.shape[0], -1)
    num_features = data.shape[1]
    for i in range(num_features):
        if any(data[1:, i] == b""):
            col = data[1:, i]
            if Counter(col).most_common(1)[0][0] == b"":
                data[1:, :][col == b"", i] = Counter(col).most_common(2)[1][0]
            else:
                data[1:, :][col == b"", i] = Counter(col).most_common(1)[0][0]
    if test:
        test_data = test_data.reshape(test_data.shape[0], -1)
        num_features = test_data.shape[1]
        for i in range(num_features):
            if any(test_data[1:, i] == b""):
                col = test_data[1:, i]
                if Counter(col).most_common(1)[0][0] == b"":
                    test_data[1:, :][col == b"", i] = Counter(col).most_common(2)[1][0]
                else:
                    test_data[1:, :][col == b"", i] = Counter(col).most_common(1)[0][0]

    # one_hot encoding
    if one_hot:
        test_file = csv.DictReader(open(path_test))
        train_file = csv.DictReader(open(path_train))
        dict_filter = lambda d, key: dict((i, d[i]) for i in d if i in set(key))
        file_dict = [dict_filter(row, key_wanted) for row in train_file] + [dict_filter(row, key_wanted) for row in
                                                                            test_file]
        V = DictVectorizer()
        X = V.fit_transform(file_dict).toarray()
        OH_features = V.get_feature_names()
        aug_data = np.vstack((data, test_data[1:]))
        dataf = aug_data[1:, [0, 2, 3, 4]].astype("float")
        new_data = np.hstack((dataf, X))

        new_features = np.hstack((data[0, [0, 2, 3, 4]], OH_features))
        train = np.vstack((new_features, new_data[0:train_size, :]))
        test = np.vstack((new_features, new_data[train_size:, :]))
        return train, test
    return data.astype("float")


if __name__ == "__main__":
    # dataset = "titanic"
    dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data       
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]
        data = np.delete(data, np.where(data[0, :] == b"ticket"), axis=1)
        test_data = np.delete(test_data, np.where(test_data[0, :] == b"ticket"), axis=1)
        train_size = data.shape[0] - 1

        key_wanted = ("sex", "fare", "cabin", "embarked")
        label = data[1:, 0]
        label = prepross(label, key_wanted=key_wanted, one_hot=False).reshape(label.shape[0], )
        process_data, process_test_data = prepross(data[:, 1:], key_wanted=key_wanted, test_data=test_data, test=True,
                                                   train_size=train_size)

        # 3.4 Performance Evaluation for titanic

        # cross_validation(process_data, label, cv=5, dataset="titanic")
        # cross_validation(process_data, label, cv=5, dataset="titanic", tree=RandomForest, max_depth=7, p=20)
        # The average training accuracy is 0.8787500000000001
        # The average validation accuracy is 0.788
        # The average training accuracy is 0.7777499999999999
        # The average validation accuracy is 0.7460000000000001

        # tree = DecisionTree(max_depth=10, feature_labels=process_data[0, 1:])
        # tree.fit(process_data[1:,:],label)
        # score = sklearn.metrics.accuracy_score(label, tree.predict(process_data[1:,:]))
        # test_pred = tree.predict(process_test_data[1:,:])
        # results_to_csv(test_pred)
        # print(score)

        # 3.6 Visualization of a shallow decision tree
        dt = DecisionTree(max_depth=3, feature_labels=process_data[0, 1:])
        dt.fit(process_data[1:, :].astype("float"), label)
        print(dt)
    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = './datasets/spam_data/spam_data.mat'
        key_wanted = ("sex", "fare", "cabin", "embarked")
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]
    ######################################################################################
    # cross_validation(X,y)
    # dt = DecisionTree(max_depth=20, feature_labels=features)
    # dt.fit(X, y)
    # dt.predict(X[0,:], verbose=True)
    # score = sklearn.metrics.accuracy_score(y,dt.predict(X))
    # print(score)
    # print("Predictions", np.sum(dt.predict(X)))
    # print("Tree structure", dt.__repr__())
    # rf = RandomForest(max_depth=20, feature_labels=features, p=6, n=20)
    # rf.fit(X,y,Randomf=True)
    # pred = np.array(rf.predict(Z))
    # results_to_csv(pred)
    # score = sklearn.metrics.accuracy_score(y, rf.predict(X))
    # print(score)
    # cross_validation(X, y, cv=5, tree=RandomForest, p=6, n=20, max_depth=20)
    ###########################################################################################

    # 3.4 Performance Evaluation for Spam/Ham
    # cross_validation(X, y, cv=5, features=features)
    # cross_validation(X, y, cv=5, tree=RandomForest, p=6, n=20, max_depth=20, features=features)
    # The average training accuracy is 0.8834587093357484
    # The average validation accuracy is 0.8284969958605481
    # The average training accuracy is 0.857743631514771
    # The average validation accuracy is 0.8377770302469656

    # 3.5 Visualization of the Decision Tree of Spam/Ham
    # dt = DecisionTree(max_depth=20, feature_labels=features)
    # dt.fit(X, y)
    # pred = dt.predict(X[-1], verbose=True)

    # depths = np.linspace(1, 40, 20)
    # train_errors = []
    # val_errors = []
    # for depth in depths:
    #     train_error, val_error = cross_validation(X, y, cv=5, features=features, max_depth=depth)
    #     train_errors.append(train_error)
    #     val_errors.append(val_error)
    # plt.plot(depths, val_errors)
    # plt.xlabel("Depth of the tree")
    # plt.ylabel("Validation Accuracy")
    # plt.title("Validation Accuracy vs Depth of the tree")
    # plt.show()
