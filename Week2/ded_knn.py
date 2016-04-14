# coding=utf-8
__author__ = 'ded'

import os
from sklearn.neighbors import KNeighborsClassifier
import sklearn.cross_validation as cv
from sklearn.cross_validation import KFold
import sklearn.preprocessing
from sklearn import svm
import numpy as np
import pandas
import pylab as pl

##########
#
# Выполните следующие шаги:
#
#
# Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale. Снова найдите оптимальное k на кросс-валидации.
#
# Какое значение k получилось оптимальным после приведения признаков к одному масштабу? Приведите ответы на вопросы 3 и 4. Помогло ли масштабирование признаков?
#
# Если ответом является нецелое число, то целую и дробную часть необходимо разграничивать точкой, например, 0.5. При необходимости округляйте дробную часть до двух знаков.
#
##########

print ('#lesson 3 - task 1')

# столбцы
# class,Alcohol,Malic.acid,Ash,Alcalinity.of.ash,Magnesium,Total.phenols,Flavanoids,Nonflavanoid.phenols,Proanthocyanins,Color.intensity,Hue,OD280-OD315,Proline
csv = pandas.read_csv( 'wine.data.txt')

# Извлеките из данных признаки и классы. Класс записан в первом столбце (три варианта), признаки — в столбцах со второго по последний.
classes = np.array(csv['class'])
data = np.array(csv.drop(['class'], axis=1))

# Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold).
# Создайте генератор разбиений, который перемешивает выборку перед формированием блоков (shuffle=True).
# Для воспроизводимости результата, создавайте генератор KFold с фиксированным параметром random_state=42.
# В качестве меры качества используйте долю верных ответов (accuracy).
# Найдите точность классификации на кросс-валидации для метода k ближайших соседей (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50.
# При каком k получилось оптимальное качество? Чему оно равно (число в интервале от 0 до 1)? Данные результаты и будут ответами на вопросы 1 и 2.
kf = KFold(classes.size, n_folds=5, shuffle=True, random_state=42)

a_max = 0.
kn_max = 0
results = []
# data = sklearn.preprocessing.scale(data)
for kn in range(1, 50, 1):
    neigh = KNeighborsClassifier(n_neighbors=kn)
    r = cv.cross_val_score(estimator=neigh, X=data, y=classes, cv=kf)
    accuracy = r.mean() # подсчет точности
    # альтернатиный подсчет точности
    # preds = neigh.predict(data[test_index])
    # accuracy = np.where(preds == classes[test_index], 1, 0).sum() / float(len(test_index))
    if accuracy > a_max:
        a_max = accuracy
        kn_max = kn
    results.append([kn, accuracy])

# ответ на задание 4
print ('answer: кол-во соседей = {}, точность = {:.2f}'.format(kn_max, a_max))

# рисую график
results = pandas.DataFrame(results, columns=["kn", "accuracy"])
pl.plot(results.kn, results.accuracy)
pl.grid(True)
pl.title("accuracy vs Kn")
pl.show()