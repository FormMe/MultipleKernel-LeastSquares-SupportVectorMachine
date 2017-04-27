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


def main(argv):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    df = read_csv(argv[0])
    X = np.array(df.drop(['y'], axis=1))
    y = np.array(df.y)
    reg_param_vals = [10 ** e for e in range(-4, 5)]
    C = reg_param_vals[rank % len(reg_param_vals)]
    kernel_set = [RBF(1e+2),RBF(1e+3),RBF(1e+4)]
    res = test_classifier(X, y, C, kernel_set)
    res = comm.gather(res, root=0)

    if rank == 0:
        res = list(reduce(lambda x, y: x + y, list(res)))
        res_df = DataFrame(data=res, columns=['C', 'CV 10Kfold score'])
        res_df.to_excel('../Results/three_kernels/'+argv[1]+'_test_res.xls', index=False)


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
        loaded = read_excel('../Results/one_kernel/s_'+sigma+'_test_res1.xls')
        loaded['sigma'] = sigma
        df.append(loaded)
    df = concat(df, ignore_index=True)
    print(df['CV 10Kfold score'].max())
    df = df[df['CV 10Kfold score'] < df['CV 10Kfold score'].max()]
    df1 = df[df['CV 10Kfold score'] > 92]
    print(df1)
    df = df.groupby(['sigma']).count()
    #df.to_csv('d.csv')
    print(df)

def pplot():
    df = read_csv('../data/test_sparse.csv')
    X = np.array(df.drop(['y'], axis=1))
    y = np.array(df.y)
    clf = MKLSSVM([RBF(1e+2),RBF(1e+3),RBF(1e+4)], C=1e+2).fit(X,y)
    #clf = svm.SVC(C=10000, gamma=1e+1, kernel='rbf').fit(X,y)
    p = clf.predict(X)
    print(accuracy_score(list(p), list(y)))
    plot_decision_regions(X,y,clf, resolution=0.5)

if __name__ == "__main__":
    #pplot()
    main(sys.argv[1:])
    #best_clf()
    # try:
    # except:
    #     print("Unexpected error:", sys.exc_info()[0])
