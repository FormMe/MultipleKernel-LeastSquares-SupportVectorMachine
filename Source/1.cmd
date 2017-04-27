@echo off
set sigma[0]=1e-1
set sigma[1]=1
for /F "tokens=2 delims==" %%s in ('set sigma[') do (mpiexec -n 9 python p.py %%s )