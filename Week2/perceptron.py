import pandas as pd
from sklearn.linear_model import Perceptron

'''
Линейные алгоритмы — распространенный класс моделей, которые отличается своей простотой и скоростью работы.
Их можно обучать за разумное время на очень больших объемах данных,
и при этом они могут работать с любыми типами признаков — вещественными, категориальными, разреженными.
 В этом задании мы предлагаем вам воспользоваться персептроном — одним из простейших вариантов линейных моделей.
'''

'''
Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
Целевая переменная записана в первом столбце, признаки — во втором и третьем.
'''
dataTest = pd.read_csv('perceptron-test.csv', names=col)
dataTrain = pd.read_csv('perceptron-train.csv', names=col)
y = dataTrain[1]
X = dataTrain.ix[:, :2]
print(y)
print("---------------\n")
print(x)
'''Обучите персептрон со стандартными параметрами и random_state=241.'''
clf = Perceptron(random_state=241)
# clf.fit()
