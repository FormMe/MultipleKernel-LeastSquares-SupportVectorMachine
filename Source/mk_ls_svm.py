import numpy
import scipy
from scipy.optimize import (minimize)
from functools import reduce


class MKLSSVM:
    def __init__(self, kernel_set, C=1.0, R=1.0, tol=1e-4, max_iter=500):
        self.C = C
        self.R = R
        self.tol = tol
        self.max_iter = max_iter
        self.kernel_set = kernel_set
        self.beta = numpy.array([1.0 / len(kernel_set) for _ in kernel_set])

    def fit(self, data, target):
        def unweighted_kernel_matrix():
            # H_vec представляет из себя вектор матриц вычисленных ядерных функций
            # взвешенная сумма этих матриц дает искомую матрицу ядер
            # значение ядер не поменяется на протяжении всего алгоритма
            # будут меняться только веса
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
            for i in range(trainSeqLen):
                for j in range(i, trainSeqLen):
                    H[i, j] *= target[i] * target[j]
                    H[j, i] *= target[j] * target[i]
                    if i == j:
                        H[i, j] += 1.0 / self.C

            d = numpy.ones(trainSeqLen)
            eta = scipy.sparse.linalg.cg(H, target)[0]
            nu = scipy.sparse.linalg.cg(H, d)[0]
            s = numpy.dot(target.T, eta)
            b = numpy.dot(eta.T, d) / s
            alpha = nu - eta * b
            return b, alpha

        def kernel_coefficient_estimation():
            def score_func(beta_vec):
                def K_sum(i):
                    weighted_kernels = []
                    for b_c, H in zip(beta_vec, self.__Hvec):
                        weighted_kernels.append(
                            b_c * numpy.asarray([y * H[i, j] for j, y in enumerate(target)], dtype=float))
                    return numpy.array(reduce(lambda l, m: l + m, weighted_kernels))

                loss_func_vec = []
                for i, y in enumerate(target):
                    weighted_kernels_sum = K_sum(i)
                    loss_func_vec.append(1.0 - y * self.b - y * numpy.dot(weighted_kernels_sum, self.alpha))

                loss_func = reduce(lambda e1, e2: e1 + e2 ** 2, loss_func_vec)
                return loss_func

            cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1.0})
            bnds = [(0.0, 1.0) for _ in self.beta]
            betaopt = minimize(score_func, self.beta, bounds=bnds, constraints=cons, method='SLSQP')
            return betaopt.x, betaopt.fun

        classes = numpy.unique(target)
        if len(classes) == 1 or len(classes) != 2:
            raise Exception('The number of classes has to be equal two')

        self.class_dict = {
            '1.0': classes[0],
            '-1.0': classes[1]}
        target = numpy.array(list(map(lambda y: 1.0 if y == classes[0] else -1.0, target)))

        self.__Xfit = data
        self.__Yfit = target
        self.__Hvec = unweighted_kernel_matrix()
        prev_score_value = 0
        prev_beta_norm = numpy.linalg.norm(self.beta)
        cur_iter = 0
        while True:
            self.b, self.alpha = lagrange_coefficient_estimation()
            self.beta, score_value = kernel_coefficient_estimation()
            # выход по количеству итераций
            if cur_iter >= self.max_iter:
                break
            # выход по невязке функции
            if abs(prev_score_value - score_value) < self.tol:
                break
            # выход по невязке нормы коэфициентов
            beta_norm = numpy.linalg.norm(self.beta)
            if abs(prev_beta_norm - beta_norm) < self.tol:
                break
            prev_score_value = score_value
            prev_beta_norm = beta_norm
            cur_iter += 1

        return self

    def predict(self, data):
        def y_prediction(z):
            def weighted_kernel(z, x):
                return sum([beta * K.compute(z, x) for beta, K in zip(self.beta, self.kernel_set)])

            support_vectors_sum = sum(
                [alpha * y * weighted_kernel(z, x) for alpha, x, y in zip(self.alpha, self.__Xfit, self.__Yfit)])
            p = support_vectors_sum + self.b
            if p == 0.0:
                p = 1.0;
            return self.class_dict[str(numpy.sign(p))]

        return [y_prediction(test_x) for test_x in data]
