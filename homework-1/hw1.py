# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
@auther Zero Void
@date   2020/03/05

使用sklearn框架完成作业内容。
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from kflod_cross_validation import KFlodCrossValidationData

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import *

class HW1:
    def __init__(self, dataset_path="homework-1/"):
        self.dataset_path = dataset_path
        self.data_frame = None
        self.X = None
        self.y = None
        self.sp = None
        self.cmp_cnt = 0
        self.est_data = {}

    def load_iris(self, show_class_cnt=False):
        self.__dataset_warn()
        self.est_data = {}
        self.data_frame = pd.read_csv(self.dataset_path + 'iris/iris.data', header=0, index_col=False, 
                    names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
        self.X = self.data_frame.loc[:, :'petal width']
        self.y = self.data_frame.loc[:, 'class']
        self.cls_cnt = self.data_frame['class'].value_counts()
        if show_class_cnt:
            self.__countplot("class")
    
    def load_adult(self, show_class_cnt=False):
        self.__dataset_warn()
        self.est_data = {}
        self.data_frame = pd.read_csv(self.dataset_path + 'adult/adult.data', header=0, index_col=False,
                            names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                                'relationship','race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
        self.data_frame = pd.get_dummies(self.data_frame.drop(columns=['fnlwgt']))
        self.X = self.data_frame.loc[:, :'native-country_ Yugoslavia']
        self.y = self.data_frame.loc[:, 'income_ >50K']
        self.cls_cnt = self.data_frame['income_ >50K'].value_counts()
        if show_class_cnt:
            self.__countplot('income_ >50K')
    
    def train(self, clf, n_splits=10):
        self.sp = StratifiedKFold(n_splits=n_splits)
        flod_data = KFlodCrossValidationData(n_splits)
        self.est_data[clf] = flod_data

        for i, (train_index, test_index) in enumerate(self.sp.split(self.X, self.y)):
            X_train = self.X.loc[train_index]
            y_train = self.y.loc[train_index]
            X_test = self.X.loc[test_index]
            y_test = self.y.loc[test_index]
            clf.fit(X_train, y_train)
            flod_data.add_data(clf, X_test, y_test)
    
    def plot_pr_roc_curve(self, clf, suptitle):
        """ 对clf评估数据绘制PR, ROC曲线 """
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(suptitle, fontsize=16)
        pr_ax = fig.add_subplot(221)
        roc_ax = fig.add_subplot(222)
        mean_pr_ax = fig.add_subplot(223)
        mean_roc_ax = fig.add_subplot(224)
        flod_data = self.est_data[clf]
        if flod_data is not None:
            flod_data.plot_pr_curve(pr_ax)
            flod_data.plot_roc_curve(roc_ax)
            flod_data.plot_mean_pr_curve(mean_pr_ax)
            flod_data.plot_mean_roc_curve(mean_roc_ax)
    
    def clf_cmp(self, clf_1, clf_2, n_splits):
        data_1 = train(clf_1)
        data_2 = train(clf_2)

    def train_sk(self, clf, n_splits=10):
        gnb_scores = cross_val_score(clf, self.X, self.y, cv=10, scoring='accuracy')
        print(gnb_scores)
        print(gnb_scores.mean())
    
    def __dataset_warn(self):
        if self.data_frame is not None:
            print("[WARN!] Reload dataset.")
    
    def __countplot(self, classname):
        ax = sns.countplot(x=classname, data=hw.data_frame)
        for p in ax.patches:
            ax.annotate("{}".format(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='center', xytext=(0,5), textcoords='offset points')


if __name__ == "__main__":
    gnb = GaussianNB()
    bnb = BernoulliNB()
    hw = HW1()
    hw.load_adult()
    gnb_data = hw.train(gnb)
    # bnb_data = hw.train(bnb)
    hw.plot_pr_roc_curve(gnb, 'GaussianNB')
    # cmp_fig = plt.figure()

    plt.show()
    #hw.train(bnb)
    #hw.train_sk(gnb)
    #print(hw.cls_cnt)

