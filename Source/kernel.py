import numpy


class Kernel:
    def compute(self, x, y):
        pass


class RBF(Kernel):
    def __init__(self, sigma):
        self._sigma = sigma

    def compute(self, x, y):
        return numpy.exp(-1.0 * numpy.linalg.norm(x - y) ** 2 / (2 * self._sigma ** 2))
