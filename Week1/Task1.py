# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

# Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.
df = pd.read_csv('titanic.csv', index_col='PassengerId')
femaleCount = df[(df.Sex == 'female')].count()
maleCount = df[(df.Sex == 'male')].count()
print(femaleCount)
print(maleCount)
