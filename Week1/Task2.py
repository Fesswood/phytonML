# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

# Какой части пассажиров удалось выжить?
# Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).
df = pd.read_csv('titanic.csv', index_col='PassengerId')
survived = df[(df.Survived == 1)].count()
sc = int(df['Survived'].count())
print('{0:.2f}'.format(float(survived[1]) / sc * 100))
