import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

'''
Линейные алгоритмы — распространенный класс моделей, которые отличается своей простотой и скоростью работы.
Их можно обучать за разумное время на очень больших объемах данных,
и при этом они могут работать с любыми типами признаков — вещественными, категориальными, разреженными.
В этом задании мы предлагаем вам воспользоваться персептроном — одним из простейших вариантов линейных моделей.
'''

'''
1. Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
Целевая переменная записана в первом столбце, признаки — во втором и третьем.
'''
# dataTest = genfromtxt('perceptron-train.csv', delimiter=',')
# dataTrain = genfromtxt('perceptron-test.csv', delimiter=',')
train = pandas.read_csv('perceptron-train.csv')
test = pandas.read_csv('perceptron-test.csv')
y = train[['class']]
X = train[['p1', 'p2']]
y_test = test[['class']]
X_test = test[['p1', 'p2']]
'''
2. Обучите персептрон со стандартными параметрами и random_state=241.
'''
clf = Perceptron(random_state=241)
clf.fit(X, y)
'''
3. Подсчитайте качество (долю правильно классифицированных объектов, accuracy)
 полученного классификатора на тестовой выборке.
'''
scores = clf.score(X_test, y_test)
print("score of simple data clf = %0.3f" % scores)
print("use metric acc = %0.3f " % accuracy_score(y_test, clf.predict(X_test)))

'''
4. Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.
'''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

'''
5. Обучите персептрон на новых выборках. Найдите долю правильных ответов на тестовой выборке.
'''
clf2 = Perceptron(random_state=241)
clf2.fit(X_train_scaled, y)
'''
6. Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее.
Это число и будет ответом на задание.
'''
normalize_scores = clf2.score(X_test_scaled, y_test)
print("score of normalize data clf = %0.3f" % normalize_scores)
print("use normalize metric acc = %0.3f " % accuracy_score(y_test, clf2.predict(X_test_scaled)))
print("differ between scores = %0.3f" % (normalize_scores - scores))
