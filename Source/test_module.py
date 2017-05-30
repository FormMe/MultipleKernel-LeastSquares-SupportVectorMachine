import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pandas import DataFrame, read_csv
from sklearn.metrics import mean_squared_error

from mk_ls_svm_lib.kernel import *
from mk_ls_svm_lib.crossvalidation import *

def f(x):
    return x**2
    #return 16704.2 - 33619.7*x + 27184.*x**2 - 11519.4*x**3 + 2777.54*x**4 - 382.819*x**5 + 28.0464*x**6 - 0.846072*x**7
    #return -1085.38 + 1985.84*x - 1356.23*x**2 + 453.007*x**3 - 79.1474*x**4 + 6.93702*x**5 - 0.240592*x**6

def model_generator(x_axis, y_axis, count):
    x1 = np.random.uniform(x_axis[0], x_axis[1], count)
    x2 = np.random.uniform(y_axis[0], y_axis[1], count)
    X = np.array(list(zip(x1, x2)))
    y = np.array(list(map(lambda x: 1.0 if f(x[0]) < x[1] else -1.0, X)))
    return X, y


def plot_decision_regions(X, y, classifier, resolution=0.1, hyperplane=False):
    # setup marker generator and color map
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # step = len(grid_X) / size
    # grid_X_local = grid_X[rank*step:rank+step]
    if hyperplane:
        # comm = MPI.COMM_WORLD
        # size = comm.Get_size()
        # rank = comm.Get_rank()

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))

        grid_X = np.array([xx1.ravel(), xx2.ravel()]).T
        Z = np.array(classifier.predict(grid_X))
        Z = Z.reshape(xx1.shape)

        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    label=cl)

    # x1 = np.linspace(x1_min + 1, x1_max - 1, 500)
    # x2 = list(map(f, x1))
    # plt.plot(x1,x2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def gen():
    X, y = model_generator((-5,5),(-10,25), 150)
    df = DataFrame(data=list(zip(X[:,0], X[:,1], y)), columns=['x1', 'x2', 'y'])
    df.to_csv('../data/test_4.csv', index=False)
    plot_decision_regions(X, y, 1)


def gen_data():
    x1 = np.linspace(0, 10, 100).tolist()
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

