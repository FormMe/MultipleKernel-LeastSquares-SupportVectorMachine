import numpy as np
from sklearn.metrics import accuracy_score

def cross_val_score(clf, X, y, n_splits=10):
    '''K-fold Cross-Validation

    :param clf: object
        The object to use to fit the data.
    :param X: array-like, shape = [n_samples, n_features]
        The data to fit. Can be, for example a list, or an array at least 2d.
    :param y: array-like, shape = [n_samples]
        The target variable to try to predict in the case of supervised learning.
    :param n_splits: int, optional (default=10)
        Number of folds. Must be at least 2.
    :return: float
        Score of the estimator for each run of the cross validation.
    '''
    data = list(zip(X, y))
    np.random.shuffle(data)
    data = np.array(data)
    test_size = int(len(X) / n_splits)
    scores = []
    for slice in range(cv):
        test_start = slice*test_size
        test_data = data[test_start:test_start+test_size]

        a = data[:test_start]
        b = data[test_start+test_size:]
        train_data = np.concatenate((a, b), axis=0)

        test_X = test_data[:, 0]
        test_y = test_data[:, 1]
        train_X = train_data[:, 0]
        train_y = train_data[:, 1]

        clf.fit(np.array(train_X),np.array(train_y))
        score = accuracy_score(list(test_y), list(clf.predict(test_X)))
        scores.append(score)
    return np.mean(scores), np.std(scores)
