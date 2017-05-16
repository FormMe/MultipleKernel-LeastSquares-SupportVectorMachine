from mk_ls_svm import *
from crossvalidation import *
from test_module import *
from kernel import *
from valid_model_estimator import *

import numpy as np
from functools import reduce
from sklearn.metrics import accuracy_score
from pandas import DataFrame, read_csv, read_excel, Series, concat
from sklearn import svm
from mpi4py import MPI
import sys
from sklearn.datasets import make_classification, make_circles, make_gaussian_quantiles, make_hastie_10_2

def mc():
    n_samples = 200
    # noisy_circles = make_circles(n_samples=n_samples, factor=.5,
    #                                       noise=.2)
    # X, y = noisy_circles
    X1, y1 = make_gaussian_quantiles(cov=2.,
                                     n_samples=200, n_features=2,
                                     n_classes=2, random_state=1)
    X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                     n_samples=300, n_features=2,
                                     n_classes=2, random_state=1)
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, - y2 + 1))
    df = DataFrame(data=list(zip(X[:, 0], X[:, 1], y)), columns=['x1', 'x2', 'y'])
    df.to_csv('../data/gaussian.csv', index=False)
    plot_decision_regions(X, y, 1, hyperplane=False)


def main(argv):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    data_file = argv[1]
    res_file = argv[2]

    df = read_csv(data_file)
    # df = df[df['iris_class'] == 'Iris-setosa']
    X = np.array(df.drop(['y'], axis=1))
    y = np.array(df['y'])
    reg_param_vals = [10 ** e for e in range(-4, 5)]
    C = reg_param_vals[rank]
    # kernel_set = [RBF(gamma) for gamma in reg_param_vals]
    kernel_set = [RBF(gamma) for gamma in reg_param_vals]#[RBF(float(argv[0]))]
    # kernel_set += [Poly(0, d) for d in range(1,4)]
    clf = MKLSSVM(kernel_set, C=C)
    score = cross_val_score(clf, X, y, cv=5)
    res = (C, score * 100)
    print(res)
    res = comm.gather(res, root=0)

    if rank == 0:
        #res = list(reduce(lambda x, y: x + y, list(res)))
        res_df = DataFrame(data=res, columns=['C', 'score'])
        res_df.to_excel(res_file, index=False)


def best_clf():

    sigmas = ['1e-4',
              '1e-3',
              '1e-2',
              '1e-1',
              '1e+0',
              '1e+1',
              '1e+2',
              '1e+3',
              '1e+4']

    df =[]
    for sigma in sigmas:
        loaded = read_excel('../Results/1k/'+sigma+'_test_3_res.xls')
        loaded['sigma'] = sigma
        df.append(loaded)
    df = concat(df, ignore_index=True)
    df = df.sort(['score'], ascending=0)
    print(df.head(10))

def pplot():
    df = read_csv('../data/noisy_circles.csv')
    X = np.array(df.drop(['y'], axis=1))
    y = np.array(df['y'])
    reg_param_vals = [10 ** e for e in range(-4, 5)]
    kernel_set = [RBF(gamma) for gamma in reg_param_vals]  # [RBF(float(argv[0]))]
    #kernel_set += [Poly(0, d) for d in range(1, 4)]
    clf = MKLSSVM(kernel_set, C=1)
    clf = clf.fit(X,y)
    # p = clf.predict(X)
    # print(accuracy_score(list(p), list(y)))
    print('beta:',clf.beta )
    plot_decision_regions(X,y,clf, resolution=0.1, hyperplane=True)

if __name__ == "__main__":
    #mc()
    pplot()
    agrv_p = ['1e+4', '../data/noisy_circles.csv', '../Results/noisy_circles/1e-1_to_1e+2_noisy_circles_res.xlsx']
    #sys.argv[1:]
    # main(sys.argv[1:])
    # best_clf()
    # gen()
    # try:
    # except:
    #     print("Unexpected error:", sys.exc_info()[0])
