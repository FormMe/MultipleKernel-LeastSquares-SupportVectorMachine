from mk_ls_svm_lib.kernel import *
from mk_ls_svm_lib.crossvalidation import *
from mk_ls_svm_lib.mk_ls_svm import *

from test_module import *

import numpy as np
from functools import reduce
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import svm
from mpi4py import MPI
import sys
from sklearn.datasets import make_classification, make_circles, make_moons
import matplotlib.pyplot as plt

def mc():
    n_samples = 500
    X, y = make_classification(n_samples=200, n_features=10, n_redundant=0, n_informative=8)
    # rng = np.random.RandomState(2)
    # X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [linearly_separable,
                # make_moons(noise=0.3, random_state=0),
                # make_circles(noise=0.2, factor=0.5, random_state=1)
                ]
    names = ['10_features', 'moons', 'circles']
    for ds, name in zip(datasets, names[:1]):
        X, y = ds
        plot_decision_regions(X, y, 1)
        data = [X[:, i] for i in range(2)]
        columns = ['x' + str(i) for i in range(2)] + ['y']
        df = pd.DataFrame(data=list(zip(*data, y)), columns=columns)
        df.to_csv('../data/' + name + '.csv', index=False)

def main(argv):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    data_file = argv[1]
    res_file = argv[2]

    df = pd.read_csv(data_file, header=None,  sep=';')
    # X = np.array(df.drop(['y'], axis=1))
    # y = np.array(df['y'])

    X = np.array(df.drop(df.columns[41], axis=1))
    y = np.array(df[41])

    reg_param_vals = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 1.5, 3, 5, 10, 20, 50, 100, 150, 200, 250, 400, 700, 1000, 5000, 10000 ]
    C = reg_param_vals[rank]
    # kernel_set = [RBF(gamma) for gamma in reg_param_vals]
    # kernel_set += [Poly(0, d) for d in range(1,4)]

    if argv[0] == 'rbf':
        kernel_set = [RBF(10 ** (float(argv[3])))]
    elif argv[0] == 'poly':
        kernel_set = [Poly(0, float(argv[3]))]
    elif argv[0] == 'setrbf':
        kernel_set = [RBF(10 ** (float(gamma))) for gamma in argv[3:]]
    elif argv[0] == 'setpoly':
        kernel_set = [Poly(0, float(d)) for d in argv[3:]]
    elif argv[0] == 'fullrbf':
        gammas = ['-4', '-3', '-2', '-1', '-0.5', '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4']
        kernel_set = [RBF(10 ** (float(gamma))) for gamma in gammas[-5:]]
    elif argv[0] == 'setpoly':
        poly = ['2', '3', '4', '5']
        kernel_set = [Poly(0, float(d)) for d in poly]

    clf = MKLSSVM(kernel_set, C=C)
    score = cross_val_score(clf, X, y, cv=20)
    res = (C, score * 100)
    #print(res)
    res = comm.gather(res, root=0)

    if rank == 0:
        res_df = pd.DataFrame(data=res, columns=['C', 'score'])
        res_df.to_excel(res_file, index=False)
        print(res_file)


def best_clf(file):
    gammas = ['-4', '-3', '-2', '-1', '-0.5', '0', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4']

    df = pd.DataFrame()
    #for gamma in gammas:
    for i in range(1,9):
        loaded = pd.read_excel('../Results/' + file + '/'+ str(i) + 'rbf_' + file + '_res.xlsx')
        loaded.index = loaded['C']
        df['RBF' + str(i)] = loaded['score']

    poly = ['2', '3', '4', '5']
    #for p in poly:
    for i in range(1,5):
        loaded = pd.read_excel('../Results/' + file + '/'+ str(i) + 'poly_' + file + '_res.xlsx')
        loaded.index = loaded['C']
        df['poly' + str(i)] = loaded['score']
    print(df)
    best_df = pd.DataFrame()

    data = []
    for column in df:
        data.append((column, df[column].idxmax(), df[column].max()))
        best_df[column] = pd.DataFrame(data=[df[column].max()])

    b = pd.DataFrame(data=data, columns=['kernel', 'C', 'score'])
    #b.to_excel('best_20f.xlsx')
    coord = df.stack().argmax()
    print(coord, df.loc[coord])

    best_df.plot(kind='bar',
                 ylim=(0, 100),
                 legend=False,
                 colormap='Paired')
    plt.ylabel('Accuracy')
    plt.xlabel('kernels')

    plt.show()
    # fig, axes = plt.subplots(nrows=3, ncols=3)
    #
    # for i in range(9):
    #     df_k = df.iloc[[i]]
    #     df_k.plot(kind='bar',
    #               legend=False,
    #               title='C = ' + str(df_k.index[0]),
    #               ylim=(0, 100),
    #               ax=axes[i / 3, i % 3],
    #               colormap = 'Paired')
    #     plt.ylabel('Accuracy')
    #     plt.xlabel('')
    #
    # for i in range(3):
    #     plt.setp([a.get_xticklabels() for a in axes[i, :]], visible=False)
    # plt.show()

    # #f, axarr = plt.subplots(3, 6)
    # i = 0
    # for key, grp in df.groupby(['kernel']):
    #     plt.plot(grp['C'], grp['score'], label=key)
    #     # axarr[i/6, i%6].plot(grp['C'], grp['score'])
    #     # axarr[i/6, i%6].set_title(key)
    #     i += 1
    # plt.show()


def pplot(file):
    df = pd.read_csv('../data/'+file+'.csv')
    X = np.array(df.drop(['y'], axis=1))
    y = np.array(df['y'])
    reg_param_vals = [10 ** e for e in range(-4, 5)]

    #set = [-4, -3, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    #set = [ -3, -1, 0, 0.5, 1, 1.5, 2]
    set = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    kernel_set =[RBF(10**g) for g in set]
    clf = MKLSSVM(kernel_set, C=3)
    clf = clf.fit(X, y)

    kw = list(zip(clf.beta, [k.display() for k in kernel_set]))
    kernel_weights = pd.DataFrame(data=kw, columns=['Weight', 'Kernel'])
    kernel_weights.index = kernel_weights['Kernel']
    del kernel_weights['Kernel']
    kernel_weights = kernel_weights.transpose()
    # kernel_weights.plot(kind='bar',
    #                     ylim=(0, 1),
    #                     legend=False,
    #                     colormap='Paired')
    # plt.ylabel('weight')
    # plt.xlabel('kernels')
    # plt.show()

    kernel_weights.plot(kind='bar',
                        ylim=(0, 1),
                        stacked=True,
                        colormap='Paired')
    plt.ylabel('weight')
    plt.xlabel('kernels')
    plt.show()

    p = clf.predict(X)
    print('accuracy:', accuracy_score(list(p), list(y)))
    #plot_decision_regions(X, y, clf, resolution=0.05, hyperplane=True)


if __name__ == "__main__":
    #  mc()
    #agrv_p = ['rbf', '../data/biodeg.csv', '../Results/noisy_circles/1e-1_to_1e+2_noisy_circles_res.xlsx', '1e+4']
    # sys.argv[1:]
     main(sys.argv[1:])
    #best_clf('20_features')
    #pplot('20_features')
    # gen()
    # try:
    # except:
    #     print("Unexpected error:", sys.exc_info()[0])
