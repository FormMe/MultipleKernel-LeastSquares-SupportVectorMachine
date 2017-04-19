import numpy
from functools import reduce

from Source import mk_ls_svm
from Source.crossvalidation import cross_val_score
from sklearn.metrics import accuracy_score
import pandas

from Source import kernel
from scipy.optimize import fmin_cg


def class_transform(y, classes):
    for i, t in enumerate(y):
        if t == classes[0]:
            y[i] = 1.0
        if t == classes[1]:
            y[i] = -1.0


import scipy
if __name__ == "__main__":
    # a = [2,4]
    # trainSeqLen = 3
    # b = [numpy.matrix(numpy.ones(shape=(trainSeqLen, trainSeqLen))),numpy.matrix(numpy.ones(shape=(trainSeqLen, trainSeqLen)))]
    # b = list(map(lambda x,y : x*y, a,b))
    # print(b)
    # for i, _ in enumerate(b):
    #     print(b[i][1])
    # print(b[:][1])
    df = pandas.read_csv('../data/iris.csv')
    df = df.drop(['petal_width'], axis=1)
    df = df.drop(['petal_length'], axis=1)
    df = df[df.iris_class != 'Iris-setosa']
    X = numpy.array(df.drop(['iris_class'], axis=1))
    y = numpy.array(df.iris_class)
    classes = numpy.unique(y)
    if len(classes) == 2:
        class_transform(y, classes)
        kernelset = [kernel.RBF(1e-1), kernel.RBF(1)]
        estimator = mk_ls_svm.MKLSSVM(kernelset)
        # estimator.fit(X,y)
        # predicted = estimator.predict(X)
        # cross_val_accuracy = accuracy_score(list(y), predicted)
        cross_val_accuracy = cross_val_score(estimator, X, y, cv=5)
        print(cross_val_accuracy)
        print(numpy.mean(cross_val_accuracy))
    else:
        print('Multiclass classification is not supproted')
