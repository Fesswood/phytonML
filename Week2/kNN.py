# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from _operator import ne

import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
col = np.arange(1, 15, 1)
print(col)
wineData = pd.read_csv('wine.data.txt', names=col)
wineClass = wineData[1]
wineDimension = wineData.ix[:, 2:]
print('WINE CLASS')
# print(wineClass[-10:])
print('WINE DIMENSION')
# print(wineDimension[-10:])
a = np.zeros(shape=5 * 50)
kf = KFold(wineClass.count(), shuffle=True, n_folds=5, random_state=42)
for x in range(1, 50):
    neigh = KNeighborsClassifier(n_neighbors=x)
    print("KNeighbors: %d " % x)
    i = x
    scores = cross_val_score(neigh, wineData, y=wineClass, scoring='accuracy', cv=kf)
    a[i] = scores.mean(axis=0);
    print('{0} and {1:0.5f}'.format(i,a[i]))
    """for train, test in kf:
        neigh.fit(wineDimension.loc[train], wineClass.loc[train])
        j += 1
        index = j + (5 * (i - 1))
        print(index)
        a[index] = neigh.score(wineDimension.loc[test], wineClass.loc[test])
        print("Accuracy: %0.5f " % a[index])"""
print("max accuracy %0.5f" % a.max(axis=0))
