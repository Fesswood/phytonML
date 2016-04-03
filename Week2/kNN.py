# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

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
result = np.empty(50)

# standardize the data attributes
standardizedWineData = preprocessing.scale(wineData)

for x in range(1, 50):
    neigh = KNeighborsClassifier(n_neighbors=x)
    scores = cross_val_score(neigh, standardizedWineData, wineClass, scoring='accuracy', cv=kf)
    print("KNeighbors: %d " % x)
    print(scores)
    print(np.mean(scores))
    result[x] = (np.mean(scores))

    # i = x
    # j = 0
    # for train, test in kf:
    #     neigh.fit(wineDimension.loc[train], wineClass.loc[train])
    #     j += 1
    #     index = j + (5 * (i - 1))
    #     print(index)
    #     a[index] = neigh.score(wineDimension.loc[test], wineClass.loc[test])
    #     print("Accuracy: %0.5f " % a[index])
resultValue = result[np.max(result)]
print("max item %0.5f" % np.max(result))
print("max value %0.5f" % resultValue)
