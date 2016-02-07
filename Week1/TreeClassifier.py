# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Данное задание основано на материалах лекций
#  по логическим методам и направлено на знакомство c решающими деревьями (Decision Trees).
# Вычислите важности признаков и найдите два признака с наибольшей важностью.
# Их названия будут ответами для данной задачи
# (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен).

df = pd.read_csv('titanic.csv', index_col='PassengerId')
treeData = df[['Pclass', 'Fare', 'Age', 'Sex']]
treeData = treeData.dropna(axis=0)
treeData['Sex'] = treeData['Sex'].map(lambda x: 1 if (x == 'male') else 0)
survived = df[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna(axis=0)['Survived']
clf = DecisionTreeClassifier(random_state=241)
clf.fit(treeData, survived)
importances = clf.feature_importances_
print(treeData[:10])
print(survived[:10])
print(importances)
