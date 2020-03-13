# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

class HW1:
    def __init__(self, dataset_path="homework-1/"):
        self.dataset_path = dataset_path
        self.cls_cnt = []
        self.data_frame = None
        self.X = None
        self.y = None
        self.sp = None

    def load_iris(self, show_class_cnt=False):
        self.__dataset_warn()
        self.data_frame = pd.read_csv(self.dataset_path + 'iris/iris.data', header=0, index_col=False, 
                    names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
        self.X = self.data_frame.loc[:, :'petal width']
        self.y = self.data_frame.loc[:, 'class']
        self.cls_cnt = self.data_frame['class'].value_counts()
        if show_class_cnt:
            self.__countplot("class")
    

    def load_adult(self, show_class_cnt=False):
        self.__dataset_warn()
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
        scores = []
        self.sp = StratifiedKFold(n_splits=n_splits)
        for i, (train_index, test_index) in enumerate(self.sp.split(self.X, self.y)):
            X_train = self.X.loc[train_index]
            y_train = self.y.loc[train_index]
            X_test = self.X.loc[test_index]
            y_test = self.y.loc[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = clf.score(X_test, y_pred)
            scores.append(score)
            print(f'\n---Flod {i+1}---\nAcc: {score}')
            self.cal_matrix(clf, y_test, y_pred)
            self.draw_pr_curve(y_test, y_pred)
            self.draw_roc_curve(y_test, y_pred)

        #print("Mean accuracy of each test:{}".format(score))
        print("Mean accuracy: {}".format(np.mean(scores)))
        plt.show()
    
    def train_sk(self, clf, n_splits=10):
        gnb_scores = cross_val_score(clf, self.X, self.y, cv=10, scoring='accuracy')
        print(gnb_scores)
        print(gnb_scores.mean())
    
    def cal_matrix(self, clf, y_test, y_pred):
        confusion_mat = confusion_matrix(y_test, y_pred)
        P = precision_score(y_test, y_pred)
        R = recall_score(y_test, y_pred)
        F1 = f1_score(y_test, y_pred)
        print("Confusion Matrix:\n{}".format(confusion_mat))
        print("Precision: {}\nRecall: {}\nF1: {}".format(P, R, F1))
        return [confusion_mat, P, R, F1]

    def draw_pr_curve(self, y_test, y_pred):
        p, r, _ = precision_recall_curve(y_test, y_pred)
        plt.subplot(211)
        plt.plot(r, p)
    
    def draw_roc_curve(self, y_test, y_pred):
        plt.subplot(212)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr)
        AUC = auc(fpr, tpr)
        print(f'AUC={AUC}')
    
    def __dataset_warn(self):
        if self.data_frame is not None:
            print("[WARN!] Reload dataset.")
    
    def __countplot(self, classname):
        ax = sns.countplot(x=classname, data=hw.data_frame)
        for p in ax.patches:
            ax.annotate("{}".format(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='center', xytext=(0,5), textcoords='offset points')
        plt.show()


if __name__ == "__main__":
    gnb = GaussianNB()
    bnb = BernoulliNB()
    hw = HW1()
    hw.load_adult()
    hw.train(gnb)
    hw.train(bnb)
    #hw.train_sk(gnb)
    #print(hw.cls_cnt)

