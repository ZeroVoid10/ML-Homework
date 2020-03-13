# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes as axes

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import *

class HW1:
    def __init__(self, dataset_path="homework-1/"):
        self.dataset_path = dataset_path
        self.cls_cnt = []
        self.data_frame = None
        self.X = None
        self.y = None
        self.sp = None
        self.cmp_cnt = 0

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
    
    def train(self, clf, suptitle, n_splits=10, cmp=False):
        scores = []
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        mean_p = 0.0
        mean_r = np.linspace(0, 1, 100)
        # self.cmp_cnt = 0 if cmp is False else self.cmp_cnt

        fig = plt.figure(figsize=(14,10))
        fig.suptitle(suptitle, fontsize=14)
        pr_ax = fig.add_subplot(221 if cmp is False else 231+3*self.cmp_cnt)
        pr_ax.set_title('P-R Curve Each Flod')
        pr_ax.set_ylabel('Precision')
        pr_ax.set_xlabel('Recall')
        roc_ax = fig.add_subplot(222 if cmp is False else 232+3*self.cmp_cnt)
        roc_ax.set_title('ROC Curve Each Flod')
        roc_ax.set_ylabel('True Positive Rate')
        roc_ax.set_xlabel('False Positive Rate')

        if cmp is False:
            mean_pr_ax = fig.add_subplot(223)
            mean_pr_ax.set_title('Mean P-R Curve')
            mean_pr_ax.set_ylabel('Precision')
            mean_pr_ax.set_xlabel('Recall')
            mean_roc_ax = fig.add_subplot(224)
            mean_roc_ax.set_title('Mean ROC Curve')
            mean_roc_ax.set_ylabel('True Positive Rate')
            mean_roc_ax.set_xlabel('False Positive Rate')

        self.sp = StratifiedKFold(n_splits=n_splits)
        for i, (train_index, test_index) in enumerate(self.sp.split(self.X, self.y)):
            X_train = self.X.loc[train_index]
            y_train = self.y.loc[train_index]
            X_test = self.X.loc[test_index]
            y_test = self.y.loc[test_index]
            clf.fit(X_train, y_train)
            y_predict_proba = clf.predict_proba(X_test)
            y_pred = clf.predict(X_test)
            score = clf.score(X_test, y_pred)
            scores.append(score)
            print(f'\n---Flod {i+1}---\nAcc: {score}')
            self.cal_matrix(clf, y_test, y_pred)
            tmp_pr = self.plot_pr_curve(y_test, y_predict_proba, pr_ax, i+1)
            mean_p += np.interp(mean_r, tmp_pr[1][::-1], tmp_pr[0][::-1])

            tmp_roc = self.plot_roc_curve(y_test, y_predict_proba, roc_ax, i+1)
            mean_tpr += np.interp(mean_fpr, tmp_roc[0], tmp_roc[1])
            mean_tpr[0] = 0.0
            # plot_precision_recall_curve(clf, X_test, y_test)
            # plot_roc_curve(clf, X_test, y_test)
            # break

        #print("Mean accuracy of each test:{}".format(score))
        print("Mean accuracy: {}".format(np.mean(scores)))
        mean_p /= n_splits
        mean_p[-1] = 0.0
        mean_p[0] = 1.0
        mean_ap = auc(mean_r, mean_p)
        mean_pr_ax.plot(mean_r, mean_p, label=f'mean P-R(mean AP = {mean_ap:.2f})')

        mean_tpr /= n_splits
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        mean_roc_ax.plot(mean_fpr, mean_tpr, label=f'mean ROC(mean AUC = {mean_auc:.2f})')
        pr_ax.legend(loc='lower left')
        roc_ax.legend(loc='lower right')
        mean_pr_ax.legend(loc='lower left')
        mean_roc_ax.legend(loc='lower right')
        return [fig, {'p':mean_p, 'r':mean_r, 'ap':mean_ap},
                {'fpr':mean_fpr, 'tpr':mean_tpr, 'auc':mean_auc}]
    
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

    def plot_pr_curve(self, y_test, y_predict_proba, ax, flod_index=0):
        p, r, _ = precision_recall_curve(y_test, y_predict_proba[:,1])
        ap = auc(r, p)
        ax.plot(r, p, lw=1, label='P-R' + (f' flod {flod_index}' if flod_index>0 else '') + f'(AP = {ap:.2f})')
        return [p, r, _]
    
    def plot_roc_curve(self, y_test, y_predict_proba, ax, flod_index=0):
        fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba[:,1])
        AUC = auc(fpr, tpr)
        print(f'AUC={AUC}')
        ax.plot(fpr, tpr, linewidth=1, label='ROC' + 
            (f' flod {flod_index}' if flod_index>0 else '') + f'(AUC = {AUC:.2f})')
        return [fpr,tpr, thresholds]
    
    def clf_cmp(self, clf_1, clf_2, n_splits):
        data_1 = train(clf_1)
        data_2 = train(clf_2)
    
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
    gnb_data = hw.train(gnb, 'GaussianNB')
    bnb_data = hw.train(bnb, "BernoulliNB")
    cmp_fig = plt.figure()

    plt.show()
    #hw.train(bnb)
    #hw.train_sk(gnb)
    #print(hw.cls_cnt)

