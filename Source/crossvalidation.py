import numpy as np
from sklearn.metrics import accuracy_score

#K-fold Cross-Validation
def cross_val_score(clf, X, y, rank, cv=10):
    data = list(zip(X, y))
    np.random.shuffle(data)
    data = np.array(data)
    test_size = int(len(X) / cv)
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

        clf.fit(train_X, train_y)
        scores.append(accuracy_score(list(test_y), clf.predict(test_X)))

    return np.mean(scores)
