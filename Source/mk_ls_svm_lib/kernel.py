import numpy as np


class RBF:
    def __init__(self, sigma):
        self._sigma = sigma

    def compute(self, x, y):
        sqr_dist = sum([(it1 - it2)**2 for it1, it2 in zip(x,y)])
        return np.e**(-1.0 * (sqr_dist ** 2) / (2 * (self._sigma ** 2)))

    def display(self):
        return 'RBF('+str(round(self._sigma, 4))+')'

class Poly:
    def __init__(self, c, d):
        self.__c = c
        self.__d = d

    def compute(self, x, y):
        return (np.dot(x,y) + self.__c)**self.__d

    def display(self):
        return 'Poly(' + str(self.__c) + ', ' + str(self.__d) + ')'
