# coding=utf-8
__author__ = 'ded'

import numpy as np
import pandas
from sklearn.metrics import roc_auc_score


def sigmoid(w, X, y):
    z = X.dot(w)
    return 1 / (1 + np.exp(-z * y))


def sigmoid_error(w, X):
    z = X.dot(w)
    return 1 / (1 + np.exp(-z))


def update_grad(X, y, w, k, C):
    w0 = w[0] + (k / y.shape[0]) * np.sum(y * X['p1'] * (1 - sigmoid(w, X, y))) - k * C * w[0]
    w1 = w[1] + (k / y.shape[0]) * np.sum(y * X['p2'] * (1 - sigmoid(w, X, y))) - k * C * w[1]
    return np.array([w0, w1])


# алгоритм логистической регрессии с градиентом
def logRegrGrad(X, y, theta=None, tol=0.01, iter_max=10000, k=0.1, C=10, verbose=False):
    if theta is None:
        w = np.zeros(X.shape[1])
    else:
        w = theta

    # поехали итерации проводить
    for iterN in range(1, iter_max):
        regul = 0.5 * C * np.sum(np.power(w, 2))
        regr = np.log(1 + np.exp(-y.dot(np.dot(X, w)))) + regul
        w_old = w
        w = update_grad(X, y, w, k, C)

        evkl = np.sqrt(np.sum(np.power(w_old - w, 2)))
        if evkl <= tol:  # проверяем сходимость
            if verbose:
                print('evkl = ', evkl)
                print('iterN = ', iterN)
            break

        if verbose:
            print('iter = {}'.format(iterN))
            print('regr = {}'.format(regr))
            print('w_old = ', w_old)
            print('w = ', w)
            print('tol = ', evkl)
            print('')
    result = (w, iterN, evkl)
    return result


csv = pandas.read_csv('data-logistik')

y = csv['class']
X = csv[['v1', 'v2']]

for c in [0, 10]:
    w, i, t = logRegrGrad(X, y, k=0.1, C=c, iter_max=10000, tol=10 ** -5)
    preds = sigmoid_error(w, X)
    r = roc_auc_score(y, preds)
    print('c: ', c)
    print('i: {}, t: {}'.format(i, t))
    print('roc_auc_score = {:0.3f}'.format(r))
    print('')
