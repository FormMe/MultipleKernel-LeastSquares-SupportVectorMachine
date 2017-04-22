import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Source.kernel import RBF

def f(z):
    data = [1,3,5,6]
    targrt =[2,6,5,7]
    alpha = [10,4,5,1]
    kernel_set = [RBF(1e-3),RBF(1e-2),RBF(1e-1),RBF(1),RBF(10),RBF(10)]
    beta = [0.5,0,0.1,0.3,0,0.1]
    b = -3
    def weighted_kernel(z, x):
        return sum([bt * K.compute(z, x) for bt, K in zip(beta, kernel_set)])

    support_vectors_sum = sum([a * y * weighted_kernel(z, x) for a, x, y in zip(alpha, data, targrt)])

    return support_vectors_sum + b

def model_generator(rng, count):
    x1 = np.random.uniform(rng[0], rng[1], count)
    x2 = np.random.uniform(rng[0], rng[1], count)
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
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    label=cl)

    data = np.linspace(x1_min, x1_max, 200)
    target = [f(x) for x in data]
    plt.plot(data, target)
    plt.show()
