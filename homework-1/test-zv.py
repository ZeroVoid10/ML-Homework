# -*- coding: utf-8 -*-
"""
@auther Zero Void
@date   2020/03/05

使用sklearn框架完成作业内容。
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

adult_frame = pd.read_csv('homework-1/adult/adult.data', header=0, index_col=False,
                     names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                           'relationship','race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
adult_frame = pd.get_dummies(adult_frame.drop(columns=['fnlwgt']))
X = adult_frame.loc[:, :'native-country_ Yugoslavia']
y = adult_frame.loc[:, 'income_ >50K']

skf = StratifiedKFold(n_splits=10)

for train, test in skf.split(X, y):
    print('train - {}  | test - {}'.format(np.bincount(y[train]), np.bincount(y[test])))
    break