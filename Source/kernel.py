import numpy


class RBF:
    def __init__(self, sigma):
        self._sigma = sigma

    def compute(self, x, y):
        sqr_dist = sum([(it1 - it2)**2 for it1, it2 in zip(x,y)])
        return numpy.e**(-1.0 * (sqr_dist ** 2) / (2 * (self._sigma ** 2)))
