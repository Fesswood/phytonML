from time import time

import numpy as np
from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.feature_extraction import text
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

'''
Для начала вам потребуется загрузить данные.
В этом задании мы воспользуемся одним из датасетов, доступных в scikit-learn'е — 20 newsgroups.
 Для этого нужно воспользоваться модулем datasets:
1 Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм"
  (инструкция приведена выше). Обратите внимание, что загрузка данных может занять несколько минут
 '''
print("Loading dataset...")
t0 = time()
newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space'])
data_samples = newsgroups.data
print("done in %0.3fs." % (time() - t0))
'''
После выполнения этого кода массив с текстами будет находиться
в поле newsgroups.data, номер класса — в поле newsgroups.target.

Одна из сложностей работы с текстовыми данными состоит в том, что для них нужно построить числовое представление.
Одним из способов нахождения такого представления является вычисление TF-IDF.
В Scikit-Learn это реализовано в классе sklearn.feature_extraction.text.TfidfVectorizer.
Преобразование обучающей выборки нужно делать с помощью функции fit_transform, тестовой — с помощью transform.

Реализация SVM-классификатора находится в классе sklearn.svm.SVC.
Веса каждого признака у обученного классификатора хранятся в поле coef_.
Чтобы понять, какому слову соответствует i-й признак,
можно воспользоваться методом get_feature_names() у TfidfVectorizer:
'''
'''
2 Вычислите TF-IDF-признаки для всех текстов.
Обратите внимание, что в этом задании мы предлагаем вам вычислить TF-IDF по всем данным.
При таком подходе получается, что признаки на обучающем множестве используют информацию из тестовой выборки —
но такая ситуация вполне законна, поскольку мы не используем значения целевой переменной из теста.
На практике нередко встречаются ситуации, когда признаки объектов тестовой выборки известны на момент обучения,
и поэтому можно ими пользоваться при обучении алгоритма.
'''
# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
vectorizer = text.TfidfVectorizer()

t0 = time()
tfidf = vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

feature_mapping = vectorizer.get_feature_names()
print(feature_mapping)

'''
3 Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5]
для SVM с линейным ядром (kernel='linear') при помощи кросс-валидации по 5 блокам.
Укажите параметр random_state=241 и для SVM, и для KFold.
В качестве меры качества используйте долю верных ответов (accuracy).
'''

grid = {'C': np.power(10.0, np.arange(start=-5.0, stop=5.0, step=1.0))}
cv = KFold(newsgroups.target.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)

'''
Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге.
'''
t0 = time()
print("fit GridSearchCV...")
gs.fit(tfidf, newsgroups.target)
print("done in %0.3fs." % (time() - t0))

# for a in gs.grid_scores_:
# a.mean_validation_score — оценка качества по кросс-валидации
# a.parameters — значения параметров


print("getting max C parameter")
print(gs.grid_scores_[np.argmax(gs.grid_scores_)])
