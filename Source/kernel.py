class Kernel:
    def compute(self, x, y):
        pass

class RBF(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma

    def compute(self, x, y):
        pass
