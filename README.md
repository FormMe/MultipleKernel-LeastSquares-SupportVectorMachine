# MultipleKernel-LeastSquares-SuportVectorMachine

**Firstly import package**
```python
from mk_ls_svm_lib as mk
```
**Create instance of classificator with list of kernels**
```python
kernel_set = [mk.kernel.RBF(10), mk.kernel.Poly(1,2)]
clf = mk.mk_ls_svm.MKLSSVM(kernel_set)
```
**Fit classificator**
```python
import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2]) 
clf = clf.fit(X,y)
```
**Predict**
```python
predicted_y = clf.predict(X)
```
**You can save your classificator into file**
```python
clf.to_pkl('my_clf.pkl')
```
**And load it**
```python
clf = mk.mk_ls_svm.load_clf_from_pkl('my_clf.pkl') 
```
**Also you can use built-in k-fold crossvalidation**
```python
score = mk.crossvalidation.cross_val_score(clf, X, y)
```
 
