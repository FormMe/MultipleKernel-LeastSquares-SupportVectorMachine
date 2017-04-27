from .mk_ls_svm import *
from .crossvalidation import *
from .test_module import *
from .kernel import *
from .valid_model_estimator import *

import numpy
from functools import reduce
from sklearn.metrics import accuracy_score
import pandas
from sklearn import svm
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        data = {'a': 7, 'b': 3.14}
        comm.send(data, dest=1, tag=11)
    elif rank == 1:
        data = comm.recv(source=0, tag=11)
    print('rank: ', rank, 'data: ', data)

    # # df = pandas.read_csv('../data/iris.csv')
    # # df = df.drop(['petal_width'], axis=1)
    # # df = df.drop(['petal_length'], axis=1)
    # # df = df[df.iris_class != 'Iris-setosa']
    # # X = numpy.array(df.drop(['iris_class'], axis=1))
    # # y = numpy.array(df.iris_class)
    # try:
    #     valid_model = init_valid_model()
    # except Exception as inst:
    #     print(inst.args)


if __name__ == "__main__":
    main()
