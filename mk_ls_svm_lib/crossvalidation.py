import numpy as np
from sklearn.metrics import accuracy_score

def cross_val_score(clf, X, y, cv=10):
    '''K-fold Cross-Validation

    :param clf: object
        Модель классификатора
    :param X: array-like, shape = [n_samples, n_features]
        Обучающая выборка. Значение факторов наблюдений.
    :param y: array-like, shape = [n_samples]
        Значения классов обучающей выборки, соответсвующие X.
    :param cv: int, optional (default=10)
        Количество разбиений кросс-валидации
    :return: float
        Точность классификации по кросс-валидациив
    '''
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

        clf.fit(np.array(train_X),np.array(train_y))
        score = accuracy_score(list(test_y), list(clf.predict(test_X)))
        scores.append(score)
    return np.mean(scores), np.std(scores)
