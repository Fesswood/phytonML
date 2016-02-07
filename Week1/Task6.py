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
miss = df['Name'].str.extract('Miss..(\w+).*').n
maredMiss = df['Name'].map(lambda x: x.startswith(' Mrs.'))
mister = df['Name'].map(lambda x: x.startswith(' Mr.'))
print(miss)
