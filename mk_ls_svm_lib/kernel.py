import numpy as np


class RBF:
    def __init__(self, sigma):
        '''
        RBF-kernel
        :param sigma: float
        '''
        self._sigma = sigma

    def compute(self, x, y):
        '''
        :param x: array-like
            First vector.
        :param y: array-like
            Second vector.
        :return: float
        '''
        sqr_dist = sum([(it1 - it2)**2 for it1, it2 in zip(x,y)])
        return np.e**(-1.0 * (sqr_dist ** 2) / (2 * (self._sigma ** 2)))

    def display(self):
        return 'RBF('+str(round(self._sigma, 4))+')'

class Poly:
    def __init__(self, c, d):
        ''' Polynomial kernel

        :param c: float
            Parameter trading off the influence of higher-order versus lower-order terms in the polynomial.
        :param d: float
            Degree.
        '''
        self.__c = c
        self.__d = d

    def compute(self, x, y):
        '''
        :param x: array-like
            First vector.
        :param y: array-like
            Second vector.
        :return: float
        '''
        return (np.dot(x,y) + self.__c)**self.__d

    def display(self):
        return 'Poly(' + str(self.__c) + ', ' + str(self.__d) + ')'
