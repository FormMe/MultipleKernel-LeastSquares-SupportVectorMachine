from Source import mk_ls_svm
from Source.crossvalidation import cross_val_score
from Source.test_module import *
from Source.kernel import RBF

import numpy
from functools import reduce
from sklearn.metrics import accuracy_score
import pandas
from sklearn import svm


if __name__ == "__main__":

    # df = pandas.read_csv('../data/iris.csv')
    # df = df.drop(['petal_width'], axis=1)
    # df = df.drop(['petal_length'], axis=1)
    # df = df[df.iris_class != 'Iris-setosa']
    # X = numpy.array(df.drop(['iris_class'], axis=1))
    # y = numpy.array(df.iris_class)

    X, y = model_generator((-5,5), 150)
    try:
        X, y = model_generator((-15, 15), 300)
        clf = svm.SVC(C=10000, gamma=1e-2)
        clf = clf.fit(X, y)
        p = clf.predict(X)
        accuracy = accuracy_score(y, p)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        plot_decision_regions(X, y, clf)
        # kernelset = [RBF(1e-3),RBF(1e-2),RBF(1e-1),RBF(1),RBF(10),RBF(10)]
        # beta = [0.5,0,0.1,0.3,0,0.1]
        # clf = mk_ls_svm.MKLSSVM(kernelset)
        # clf.fit(X,y)
        # predicted = clf.predict(X)
        # accuracy = accuracy_score(y, predicted)
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))
        # test_module.plot_decision_regions(X, y, clf, resolution=0.5)
        # predicted = estimator.predict(X)
        # cross_val_accuracy = accuracy_score(list(y), predicted)
        # cross_val_accuracy = cross_val_score(clf, X, y, cv=5)
        # print(cross_val_accuracy)
        # print(numpy.mean(cross_val_accuracy))
    except Exception as inst:
        print(inst.args)
