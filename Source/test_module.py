import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pandas import DataFrame, read_csv
from sklearn.metrics import mean_squared_error

from kernel import RBF
from crossvalidation import *
from mk_ls_svm import MKLSSVM

def f(x):
    return -723.527 + 1503.5*x - 1098.49*x**2 + 382.18*x**3 - 68.5661*x**4 + 6.12013*x**5 - 0.215064*x**6

def model_generator(x_axis, y_axis, count):
    x1 = np.random.uniform(x_axis[0], x_axis[1], count)
    x2 = np.random.uniform(y_axis[0], y_axis[1], count)
    X = np.array(list(zip(x1, x2)))
    y = np.array(list(map(lambda x: 1.0 if f(x[0]) < x[1] else -1.0, X)))
    return X, y


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    #                        np.arange(x2_min, x2_max, resolution))
    #
    # Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    #
    # Z = Z.reshape(xx1.shape)
    # plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # plt.xlim(xx1.min(), xx1.max())
    # plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    label=cl)

    x1 = np.linspace(x1_min, x1_max, 500)
    x2 = list(map(f, x1))
    plt.plot(x1,x2)

    plt.show()


def gen_data():
    x1 = np.linspace(2, 8, 100).tolist()
    x2 = list(map(f, x1))
    dx1 = list(map(lambda x: x - 5.0, x2))
    dx2 = list(map(lambda x: x + 5.0, x2))
    x1 = x1 + x1
    x2 = dx1 + dx2
    X = np.array(list(zip(x1, x2)))
    y = np.array(list(map(lambda x: 1.0 if f(x[0]) < x[1] else -1.0, X)))

    df = DataFrame(data=list(zip(x1, x2, y)), columns=['x1', 'x2', 'y'])
    print(df.head())
    df.to_csv('../data/test.csv', index=False)

def test_one_kernel_classifier(X, y, sigma):
    kernel_set = [RBF(sigma)]
    reg_param_vals = [10**e for e in range(-4, 5)]

    res = []
    for C in reg_param_vals:
        for R in reg_param_vals:
            clf = MKLSSVM(kernel_set, C=C, R=R)
            score = np.mean(cross_val_score(clf, X, y))
            res.append((sigma,C,R,score))
    return DataFrame(data=res, columns=['sigma','C','R','CV 10Kfold score'])
