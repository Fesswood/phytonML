# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

# Коррелируют ли число братьев/сестер
# с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
df = pd.read_csv('titanic.csv', index_col='PassengerId')
corr = df['SibSp'].corr(df['Parch'], method='pearson')
print(corr)
