# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

# Какую долю пассажиры первого класса составляли среди всех пассажиров?
#  Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).
df = pd.read_csv('titanic.csv', index_col='PassengerId')
firstClassPassengers = df[(df.Pclass == 1)].count()
allLosers = int(df['Pclass'].count())
print('{0:.2f}'.format(float(firstClassPassengers[2]) / allLosers * 100))
