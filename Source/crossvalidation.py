import numpy
from sklearn.metrics import accuracy_score

#K-fold Cross-Validation
def cross_val_score(estimator, X, y, cv=10):
    data = list(zip(X, y))
    numpy.random.shuffle(data)
    data = numpy.array(data)
    test_size = int(len(X) / cv)
    scores = []
    for test_data_start in range(cv):
        test_data = data[test_data_start:test_data_start+test_size]
        train_data = numpy.array([x for x in data if x not in test_data])

        test_X = test_data[:, 0]
        test_y = test_data[:, 1]
        train_X = train_data[:, 0]
        train_y = train_data[:, 1]

        estimator.fit(train_X, train_y)
        scores.append(accuracy_score(list(test_y), estimator.predict(test_X)))
    return scores
