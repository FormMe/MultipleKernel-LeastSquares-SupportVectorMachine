from mk_ls_svm import *
from crossvalidation import *
from test_module import *
from kernel import *
from valid_model_estimator import *

import numpy as np
from functools import reduce
from sklearn.metrics import accuracy_score
from pandas import DataFrame, read_csv
from sklearn import svm
from mpi4py import MPI
import sys

def main(argv):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    df = read_csv('../data/test.csv')
    X = np.array(df.drop(['y'], axis=1))
    y = np.array(df.y)

    reg_param_vals = [10**e for e in range(-4, 5)]
    sigma = float(argv[0])
    C = reg_param_vals[rank]
    kernel_set = [RBF(sigma)]
    res = []
    for R in reg_param_vals:
        clf = MKLSSVM(kernel_set, C=C, R=R)
        score = np.mean(cross_val_score(clf, X, y))
        res.append((sigma, C, R, score*100))

    res = comm.gather(res, root=0)
    if rank == 0:
        res = list(reduce(lambda x,y:x+y, list(res)))
        res_df = DataFrame(data=res,  columns=['sigma','C','R','CV 10Kfold score'])
        # res = test_one_kernel_classifier(X, y, sigma])
        res_df.to_excel('test_'+argv[0]+'_res.xls', index=False)

    # try:
    #     valid_model = init_valid_model()
    # except Exception as inst:
    #     print(inst.args)


if __name__ == "__main__":
    main(sys.argv[1:])
