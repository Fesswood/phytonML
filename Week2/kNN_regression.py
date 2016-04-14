# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

# Мы будем использовать в данном задании набор данных Boston,
# где нужно предсказать стоимость жилья на основе различных характеристик расположения
# (загрязненность воздуха, близость к дорогам и т.д.).
# Подробнее о признаках можно почитать по адресу https://archive.ics.uci.edu/ml/datasets/Housing

# Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
# Результатом вызова данной функции является объект,
# у которого признаки записаны в поле data, а целевой вектор — в поле target.
streetData = load_boston();

# Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale
scaledData = scale(streetData.data)

# Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом,
# чтобы всего было протестировано 200 вариантов (используйте функцию numpy.linspace).
steps = np.linspace(start=1, stop=10, num=200)

# Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации по 5 блокам с random_state = 42,
# не забудьте включить перемешивание выборки (shuffle=True).
kf = KFold(len(streetData.target), shuffle=True, n_folds=5, random_state=42)
result = np.empty(len(steps))

for x in range(0, len(steps)):
    # Используйте KNeighborsRegressor с n_neighbors=5
    # и weights='distance' — данный параметр добавляет в алгоритм веса, зависящие от расстояния до ближайших соседей.
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', p=steps[x], metric='minkowski')
    # В качестве метрики качества используйте среднеквадратичную ошибку
    # ( параметр scoring='mean_squared_error' у cross_val_score).

    scores = cross_val_score(neigh, X=scaledData, y=streetData.target, scoring='mean_squared_error', cv=kf)
#    print("p %0.5f" % steps[x])
#    print("KNeighbors: %d " % x)
#    print(scores)
#    print(np.mean(scores))
    result[x] = (np.mean(scores))

resultValue = result[np.argmax(result)]

print("max item %d" % np.argmax(result))
print("max p %0.5f" % steps[np.argmax(result)])
print("max value %0.5f" % resultValue)
