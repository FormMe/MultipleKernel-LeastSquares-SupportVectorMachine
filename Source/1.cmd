@echo off

mpiexec -n 12 python main.py 1e-4 ../data/test_sparse.csv s
mpiexec -n 12 python main.py 1e-3 ../data/test_sparse.csv s
mpiexec -n 12 python main.py 1e-2 ../data/test_sparse.csv s
mpiexec -n 12 python main.py 1e-1 ../data/test_sparse.csv s
mpiexec -n 12 python main.py 1e+0 ../data/test_sparse.csv s
mpiexec -n 12 python main.py 1e+1 ../data/test_sparse.csv s
mpiexec -n 12 python main.py 1e+2 ../data/test_sparse.csv s
mpiexec -n 12 python main.py 1e+3 ../data/test_sparse.csv s
mpiexec -n 12 python main.py 1e+4 ../data/test_sparse.csv s
mpiexec -n 12 python main.py 1e-4 ../data/test.csv d
mpiexec -n 12 python main.py 1e-3 ../data/test.csv d
mpiexec -n 12 python main.py 1e-2 ../data/test.csv d
mpiexec -n 12 python main.py 1e-1 ../data/test.csv d
mpiexec -n 12 python main.py 1e+0 ../data/test.csv d
mpiexec -n 12 python main.py 1e+1 ../data/test.csv d
mpiexec -n 12 python main.py 1e+2 ../data/test.csv d
mpiexec -n 12 python main.py 1e+3 ../data/test.csv d
mpiexec -n 12 python main.py 1e+4 ../data/test.csv d