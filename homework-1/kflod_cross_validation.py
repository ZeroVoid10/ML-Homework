# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from sklearn.metrics import *
import numpy as np
import matplotlib.pyplot as plt

class KFlodCrossValidationData:
    def __init__(self, n_splits):
        self.n_splits = n_splits
        self.confusion_mat = []
        self.P = []
        self.R = []
        self.F1 = []
        self.pr_thr = []
        self.ap = []
        self.score = []
        self.fpr_tpr_thr = []
        self.roc = []
        self.auc = []
        self.predict_proba = []
        self.pred = []
        self.mean_p = None
        self.mean_r = np.linspace(0, 1, 100)
        self.mean_ap = None
        self.mean_tpr = None
        self.mean_fpr = np.linspace(0, 1, 100)
        self.mean_auc = None
    
    def add_data(self, clf, X, y):
        """ 添加分类方法评估数据
        Args:
            clf: sklearn库中分类器, 需要满足实现predict_proba,predict,score函数
                且需要完成训练
            X: Test Samples
            y: True labels for X
        """
        self.predict_proba.append(clf.predict_proba(X))
        self.pred.append(clf.predict(X))
        self.score.append(clf.score(X, y))
        self.__cal_matrix(clf, y)
        self.__pr_roc_curve(y)
    
    def plot_pr_curve(self, ax, index=None, fliter=None, mean=True):
        """ 在ax上绘制PR曲线, 并计算平均值
        Args:
            ax: matplotlib 中的 axes
                绘制RP曲线画布
            index: positive int or None
                非None, 绘制index折的PR曲线
            fliter: list or None
                绘制list中标记第几折的PR曲线
            mean: True or False
                在index和fliter为None时是否计算均值
        """
        ax.set_title('P-R Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        if index is None and fliter is None:
            if mean:
                self.mean_p = 0.0
            for i in range(self.n_splits):
                pr = self.pr_thr[i]
                ax.plot(pr[1], pr[0], lw=1, label='P-R' + 
                        f' flod {i+1}' + f'(AP = {self.ap[i]:.2f})')
                if mean:
                    self.mean_p += np.interp(self.mean_r, pr[1][::-1],pr[0][::-1])
            if mean:
                self.mean_p /= self.n_splits
                self.mean_p[-1]  = 0.0
                self.mean_p[0] = 1.0
                self.mean_ap = auc(self.mean_r, self.mean_p)
        ax.legend(loc='lower left')
    
    def plot_roc_curve(self, ax, index=None, fliter=None, mean=True):
        """ 在ax上绘制ROC曲线, 并计算平均值
        Args:
            ax: matplotlib 中的 axes
                绘制RP曲线画布
            index: positive int or None
                非None, 绘制index折的PR曲线
            fliter: list or None
                绘制list中标记第几折的PR曲线
            mean: True or False
                在index和fliter为None时是否计算均值
        """
        ax.set_title('ROC Curve')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        if index is None and fliter is None:
            if mean:
                self.mean_tpr = 0.0
            for i in range(self.n_splits):
                tmp = self.fpr_tpr_thr[i]
                ax.plot(tmp[0], tmp[1], lw=1, label='ROC' + 
                        f' flod {i+1}' + f'(AUC = {self.auc[i]:.2f})')
                if mean:
                    self.mean_tpr += np.interp(self.mean_fpr, tmp[0], tmp[1])
            if mean:
                self.mean_tpr /= self.n_splits
                self.mean_tpr[-1] = 1.0
                self.mean_tpr[0] = 0.0
                self.mean_auc = auc(self.mean_fpr, self.mean_fpr)
        ax.legend(loc='lower right')

    def plot_mean_pr_curve(self, ax, title='Mean P-R Curve'):
        """ ax画布上绘制mean pr 曲线 """
        ax.set_title(title)
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        if self.mean_p is None:
            for i in range(self.n_splits):
                pr = self.pr_thr[i]
                self.mean_p += np.interp(mean_r, pr[1][::-1],pr[0][::-1])
            self.mean_p /= self.n_splits
            self.mean_p[-1]  = 0.0
            self.mean_p[0] = 1.0
            self.mean_ap = auc(self.mean_r, self.mean_p)
        ax.plot(self.mean_r, self.mean_p, 
                    label=f'mean P-R(mean AP = {self.mean_ap:.2f})')
        ax.legend(loc='lower left')

    def plot_mean_roc_curve(self, ax, title='Mean ROC Curve'):
        """ ax画布上绘制mean roc曲线 """
        ax.set_title(title)
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        if self.mean_tpr is None:
            for i in range(self.n_splits):
                tmp = self.fpr_tpr_thr[i]
                self.mean_tpr += np.interp(self.mean_fpr, tmp[0], tmp[1])
            self.mean_tpr /= self.n_splits
            self.mean_tpr[-1] = 1.0
            self.mean_tpr[0] = 0.0
            self.mean_auc = auc(self.mean_fpr, self.mean_fpr)
        ax.plot(self.mean_fpr, self.mean_tpr, 
                    label=f'mean ROC(mean AUC = {self.mean_auc:.2f})')
        ax.legend(loc='lower right')

    def __cal_matrix(self, clf, y):
        """ 计算混淆矩阵,P,R,F1
            Must call after get new pred and predict_proba data
        Args:
            clf: sklearn 分类器
            y: True labels for X
        """
        self.confusion_mat.append(confusion_matrix(y, self.pred[-1]))
        self.P.append(precision_score(y, self.pred[-1]))
        self.R.append(recall_score(y, self.pred[-1]))
        self.F1.append(f1_score(y, self.pred[-1]))
    
    def __pr_roc_curve(self, y):
        """ 计算PR ROC曲线绘图数据 
            Must call after get new pred and predict_proba data
        """
        # return P R Thr
        tmp = precision_recall_curve(y, self.predict_proba[-1][:,1])
        self.pr_thr.append(tmp)
        self.ap.append(auc(tmp[1], tmp[0]))

        # return fpr tpr thr
        tmp = roc_curve(y, self.predict_proba[-1][:,1])
        self.fpr_tpr_thr.append(tmp)
        self.auc.append(auc(tmp[0], tmp[1]))
    