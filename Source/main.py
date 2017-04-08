from Source.mk_ls_svm import MKLSSVM
from Source import crossvalidation
from Source import kernel
if __name__ == "__main__":
    kernelset = [kernel.RBF(1), kernel.RBF(10)]
    svm = MKLSSVM(kernelset)
    X = [[]]
    y = []
    svm.fit(X, y)
    svm.predict(X)
    print(655)
