# -*- coding: utf-8 -*-
"""
@auther Zero Void
@date   2020/03/05

使用sklearn框架完成作业内容。
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn import svm

"""
adult_frame = pd.read_csv('homework-1/adult/adult.data', header=0, index_col=False,
                     names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                           'relationship','race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
adult_frame = pd.get_dummies(adult_frame.drop(columns=['fnlwgt']))
X = adult_frame.loc[:, :'native-country_ Yugoslavia']
y = adult_frame.loc[:, 'income_ >50K']
"""
iris_frame = pd.read_csv('homework-1/iris/iris.data', header=0, index_col=False, 
                    names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
#iris_frame = pd.get_dummies(iris_frame)
X = iris_frame.loc[:, :'petal width']
y = iris_frame.loc[:, 'class']

skf = StratifiedKFold(n_splits=10)
svm_clf = svm.SVC()

for train_index, test_index in skf.split(X, y):
    X_train = X.loc[train_index]
    y_train = y.loc[train_index]
    svm_clf.fit(X_train, y_train)
    print(X.loc[test_index[0]])
    print(svm_clf.predict([X.loc[test_index[0]]]))
    print(y.loc[test_index[0]])
    break