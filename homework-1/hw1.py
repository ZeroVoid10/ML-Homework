# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
@auther fsy,zx,syj,Zero Void(lsx)
@date   2020/03/05

使用sklearn框架完成作业内容。
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt
from kflod_cross_validation import KFlodCrossValidationData

from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

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
        self.X = self.X.values
        self.y = self.y.values
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
        self.n_splits = n_splits
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
        self.plot_confusion_mat(clf, pre_title=type(clf).__name__ + ' ')
        flod_data.print_mat()
        
        return self
    
    def plot_pr_roc_curve(self, clf, suptitle):
        """ 对clf评估数据绘制PR, ROC曲线 """
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(suptitle, fontsize=16)
        pr_ax = fig.add_subplot(221)
        roc_ax = fig.add_subplot(222)
        mean_pr_ax = fig.add_subplot(223)
        mean_roc_ax = fig.add_subplot(224)
        self.__plot_pr_roc_curve(clf, [pr_ax, roc_ax, mean_pr_ax, mean_roc_ax])
        plt.show()

    
    def clf_cmp(self, clfs, n_splits=10):
        """ 对clfs中分类器绘图比较 """
        # Train
        length = len(clfs)
        for res in map(self.train, clfs, [n_splits]*n_splits):
            pass

        # Plot PR ROC curve
        if length == 1:
            self.plot_pr_roc_curve(clfs[0], type(clfs[0]).__name__)
        elif length == 2:
            fig = plt.figure(figsize=(18, 10))
            fig.suptitle(type(clfs[0]).__name__ + ' & ' 
                        + type(clfs[1]).__name__ + ' CMP', fontsize=16)
            mean_pr_ax = fig.add_subplot(233)
            mean_roc_ax = fig.add_subplot(236)
            self.__plot_pr_roc_curve(clfs[0], 
                            [fig.add_subplot(231), fig.add_subplot(232), 
                            mean_pr_ax, mean_roc_ax],
                            type(clfs[0]).__name__ + ' ')
            self.__plot_pr_roc_curve(clfs[1], 
                            [fig.add_subplot(234), fig.add_subplot(235), 
                            mean_pr_ax, mean_roc_ax],
                            type(clfs[1]).__name__ + ' ')
            # Paired t-test
            self.paired_ttest(clfs[0], clfs[1])
        else:
            fig = plt.figure(figsize=(14, 5))
            fig.suptitle('CMP', fontsize=16)
            pr_ax = fig.add_subplot(211)
            roc_ax = fig.add_subplot(212)
            for clf in clfs:
                if clf in self.est_data:
                    flod_data = self.est_data[clf]
                    flod_data.plot_mean_pr_curve(pr_ax, type(clf).__name__ + ' ')
                    flod_data.plot_mean_roc_curve(roc_ax, type(clf).__name__ + ' ')
        
    def plot_confusion_mat(self, clf, fig=None, pre_title=''):
        if clf in self.est_data:
            ax_list = []
            if fig is None:
                fig = plt.figure(figsize=(15,10))
                # fig = plt.figure()
            fig.suptitle(pre_title + 'Confusion Matrix', fontsize=16)
            for i in range(self.n_splits):
                ax_list.append(fig.add_subplot(3, 4, i+1))
            ax_list.append(fig.add_subplot(3,4, 11))
            ax_list.append(fig.add_subplot(3,4, 12))
            self.est_data[clf].plot_confusion_mat(ax_list)
            fig.tight_layout()

    def paired_ttest(self, clf_1, clf_2):
        if clf_1 in self.est_data and clf_2 in self.est_data:
            statistic, pvalue = ttest_rel(self.est_data[clf_1].score,
                                          self.est_data[clf_2].score)
            print('\n\n=====Paired T-Test=====')
            print(f'statisitc = {statistic}\npvalue = {pvalue}')

    def train_sk(self, clf, n_splits=10):
        gnb_scores = cross_val_score(clf, self.X, self.y, cv=10, scoring='accuracy')
        print(gnb_scores)
        print(gnb_scores.mean())
    
    def __plot_pr_roc_curve(self, clf, ax_list, pre_title=''):
        if clf in self.est_data:
            flod_data = self.est_data[clf]
            flod_data.plot_all(ax_list, pre_title)
    
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
    # hw.train(gnb)
    hw.clf_cmp([gnb, bnb])
    # hw.plot_confusion_mat(gnb)
    plt.show()
    # gnb_data = hw.train(gnb)
    # bnb_data = hw.train(bnb)
    # hw.plot_pr_roc_curve(gnb, 'GaussianNB')
    # cmp_fig = plt.figure()

    #hw.train(bnb)
    #hw.train_sk(gnb)
    #print(hw.cls_cnt)

