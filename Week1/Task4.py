# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd

# Какого возраста были пассажиры?
# Посчитайте среднее и медиану возраста пассажиров.
# В качестве ответа приведите два числа через пробел
df = pd.read_csv('titanic.csv', index_col='PassengerId')
age_series = df['Age'].dropna()

avg = np.mean(age_series)
med = np.median(age_series)
print('{0:.2f} {1:.2f}'.format(avg, med))

averageAge = age_series.mean(axis=0)
median = age_series.median(axis=0)
print('{0:.2f} {0:.2f}'.format(float(averageAge), float(median)))
