import numpy


class RBF:
    def __init__(self, sigma):
        self._sigma = sigma

    def compute(self, x, y):
        return numpy.exp(-1.0 * numpy.linalg.norm(x - y) ** 2 / (2 * self._sigma ** 2))
