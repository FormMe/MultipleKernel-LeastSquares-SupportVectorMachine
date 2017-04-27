import numpy
import scipy
from scipy.optimize import (minimize)
from functools import reduce


class ValidModel:
    def __init__(self, kernel_set, beta, C=1.0):
        self.C = C
        self.kernel_set = kernel_set
        self.beta = beta

    def fit(self, data, target):
        def unweighted_kernel_matrix():
            trainSeqLen = len(target)
            H_vec = []
            for K in self.kernel_set:
                H = numpy.matrix(numpy.zeros(shape=(trainSeqLen, trainSeqLen)))
                for i in range(trainSeqLen):
                    for j in range(i, trainSeqLen):
                        val = K.compute(data[i], data[j])
                        H[i, j] = val
                        H[j, i] = val
                H_vec.append(H)
            return H_vec

        # Large Scale Algorithm
        def lagrange_coefficient_estimation():
            trainSeqLen = len(target)
            weighted_H = map(lambda h, beta: h * beta, self.__Hvec, self.beta)
            H = reduce(lambda p_h, h: p_h + h, weighted_H)
            H = H + numpy.diag([1.0 / self.C]*len(target))

            d = numpy.ones(trainSeqLen)
            eta = scipy.sparse.linalg.cg(H, target)[0]
            nu = scipy.sparse.linalg.cg(H, d)[0]
            s = numpy.dot(target.T, eta)
            if abs(s) < 1e-20:
                s = 1e-20
            b = numpy.dot(eta.T, d) / s
            alpha = nu - eta * b
            return b, alpha


        self.__Xfit = data
        self.__Yfit = target
        self.__Hvec = unweighted_kernel_matrix()
        self.b, self.alpha = lagrange_coefficient_estimation()

        return self

    def predict(self, data):
        def y_prediction(z):
            def weighted_kernel(z, x):
                return sum([beta * K.compute(z, x) for beta, K in zip(self.beta, self.kernel_set)])

            return sum([alpha * weighted_kernel(z, x) for alpha, x in zip(self.alpha, self.__Xfit)]) + self.b

        return [y_prediction(test_x) for test_x in data]

