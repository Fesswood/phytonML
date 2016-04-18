import numpy as np
import pandas
from sklearn.metrics import roc_auc_score


def dist_euclide(x, y):
    return np.sqrt(np.sum(np.power(np.asarray(x) - np.asarray(y), 2)))


def summa(X, y, w, index_w):
    summ = 0
    i = 0
    for xi in X:
        summ += y.item(i) * xi[index_w] * (1 - sigmoid(w, xi, y.item(i)))
        i += 1
    return summ.item(0)


def sigmoid(w, xi, y):
    return 1 / (1 + np.exp((w[0] * xi[0] + w[0] * xi[1]) * -y))


def sigmoid_error(w, xi):
    return 1 / (1 + np.exp(-w[0] * xi[0] - w[0] * xi[1]))


def logit_regression(X, y, C=10, E=10 ** -5, k=0.01, iter_count=10000, need_logging=False):
    l = float(X.size)
    w = [0, 0]
    for i in range(0, iter_count):

        w_prev = [w[0] + 0, w[1] + 0]
        w[0] = w[0] + float(k) / l * summa(X, y, w, 0) - k * w[0] * C
        w[1] = w[1] + float(k) / l * summa(X, y, w, 1) - k * w[1] * C
        dist = dist_euclide(w_prev, w)
        if need_logging:
            print("iter %d" % i)
            print("dist %0.5f" % dist)
        if dist <= E:
            if need_logging:
                print("stop logit regression on iter № %d" % i)
            return w
    return w


data = pandas.read_csv('data-logistik', )
y = np.asarray(data[['class']])
X = np.asarray(data[['v1', 'v2']])

for c in range(0, 11):
    w_aim = logit_regression(X, y, C=c)
    p = []
    for x in X:
        p.append(sigmoid_error(w_aim, x))
    print("best w:")
    print(w_aim)
    print('roc auc score:')
    print(roc_auc_score(y, p))
