# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from sklearn.metrics import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.validation import column_or_1d

class KFlodCrossValidationData:
    def __init__(self, n_splits):
        self.clf_name = ''
        self.n_splits = n_splits
        self.confusion_mat = []
        self.sum_conf_mat = np.array([[0,0],[0,0]])
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
    
    def add_data(self, clf, X, y, sk=True):
        """ 添加分类方法评估数据
        Args:
            clf: sklearn库中分类器, 需要满足实现predict_proba,predict,score函数
                且需要完成训练
            X: Test Samples
            y: True labels for X
        """
        self.clf_name = type(clf).__name__ + ' '
        self.predict_proba.append(clf.predict_proba(X))
        self.pred.append(clf.predict(X))
        self.score.append(clf.score(X, y))
        self.__cal_matrix(clf, y)
        self.__pr_roc_curve(y, sk=sk)
    
    def print_mat(self, print_conf=False):
        print(f'\n\n====={self.clf_name}Confusion Matrix & P & R & F1=====')
        for i,(mat,p,r,f1) in enumerate(zip(self.confusion_mat,
                                self.P, self.R, self.F1)):
            print(f'\n---Flod {i+1}--')
            if print_conf:
                print(f'Confusion Mat:\n{mat}')
            print(f'P = {p}')
            print(f'R = {r}')
            print(f'F1 = {f1}')
        print('\n---Mean P & R & F1---')
        print(f'Mean P = {np.mean(self.P)}')
        print(f'Mean R = {np.mean(self.R)}')
        print(f'Mean F1 = {np.mean(self.F1)}')
        if print_conf:
            print('\n---Sum/Mean Confusion Matrix---')
            print(f'Sum Confusion Matrix = \n{self.sum_conf_mat}')
            print(f'Mean Confusion Matrix = \n{self.sum_conf_mat/self.n_splits}')

    def plot_confusion_mat(self, ax_list, pre_title=''):
        for i,(mat,ax) in enumerate(zip(self.confusion_mat, ax_list)):
            vis = ConfusionMatrixDisplay(mat, '01')
            vis.plot(ax=ax, values_format='d')
            ax.set_title(f'flod {i+1}')
        if len(ax_list) == self.n_splits + 2:
            vis = ConfusionMatrixDisplay(self.sum_conf_mat,'01')
            vis.plot(ax=ax_list[-2], values_format='d')
            ax_list[-2].set_title('Sum Confusion Matrix')
            vis = ConfusionMatrixDisplay(self.sum_conf_mat/self.n_splits,'01')
            vis.plot(ax=ax_list[-1], values_format='.2f')
            ax_list[-1].set_title('Mean Confusion Matrix')
    
    def plot_pr_curve(self, ax, pre_title='', index=None, fliter=None, mean=True):
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
        ax.set_title(pre_title + 'P-R Curve')
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
    
    def plot_roc_curve(self, ax, pre_title='',index=None, fliter=None, mean=True):
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
        ax.set_title(pre_title + 'ROC Curve')
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
                self.mean_auc = auc(self.mean_fpr, self.mean_tpr)
        ax.legend(loc='lower right')

    def plot_mean_pr_curve(self, ax, pre_title=''):
        """ ax画布上绘制mean pr 曲线 """
        ax.set_title('Mean P-R Curve')
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
                    label=f'{pre_title}mean P-R(mean AP = {self.mean_ap:.2f})')
        ax.legend(loc='lower left')

    def plot_mean_roc_curve(self, ax, pre_title=''):
        """ ax画布上绘制mean roc曲线 """
        ax.set_title('Mean ROC Curve')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        if self.mean_tpr is None:
            for i in range(self.n_splits):
                tmp = self.fpr_tpr_thr[i]
                self.mean_tpr += np.interp(self.mean_fpr, tmp[0], tmp[1])
            self.mean_tpr /= self.n_splits
            self.mean_tpr[-1] = 1.0
            self.mean_tpr[0] = 0.0
            self.mean_auc = auc(self.mean_fpr, self.mean_tpr)
        ax.plot(self.mean_fpr, self.mean_tpr, 
                    label=f'{pre_title}mean ROC(mean AUC = {self.mean_auc:.2f})')
        ax.legend(loc='lower right')
    
    def plot_all(self, ax_list, pre_title=''):
        self.plot_pr_curve(ax_list[0], pre_title=pre_title)
        self.plot_roc_curve(ax_list[1], pre_title=pre_title)
        self.plot_mean_pr_curve(ax_list[2], pre_title=pre_title)
        self.plot_mean_roc_curve(ax_list[3], pre_title=pre_title)

    def __cal_matrix(self, clf, y):
        """ 计算混淆矩阵,P,R,F1
            Must call after get new pred and predict_proba data
        Args:
            clf: sklearn 分类器
            y: True labels for X
        """
        self.confusion_mat.append(confusion_matrix(y, self.pred[-1]))
        self.sum_conf_mat += self.confusion_mat[-1]
        self.P.append(precision_score(y, self.pred[-1]))
        self.R.append(recall_score(y, self.pred[-1]))
        self.F1.append(f1_score(y, self.pred[-1]))
    
    def __pr_roc_curve(self, y, sk=True):
        """ 计算PR ROC曲线绘图数据 
            Must call after get new pred and predict_proba data
        """
        # return P R Thr
        tmp = precision_recall_curve(y, self.predict_proba[-1][:,1]) if sk else self.__precision_recall_curve(y, self.predict_proba[-1][:,1])
        self.pr_thr.append(tmp)
        self.ap.append(auc(tmp[1], tmp[0]))

        # return fpr tpr thr
        tmp = roc_curve(y, self.predict_proba[-1][:,1])
        self.fpr_tpr_thr.append(tmp)
        self.auc.append(auc(tmp[0], tmp[1]))
    
    def __precision_recall_curve(self, y_true, probas_pred):
        """ 参考skkearn实现 """
        p = []
        r = []
        thr = []
        # 获取可作为阈值index
        y_true = column_or_1d(y_true)
        probas_pred = column_or_1d(probas_pred)
        desc_score_indices = np.argsort(probas_pred, kind="mergesort")[::-1]
        probas_pred = probas_pred[desc_score_indices]
        y_true = y_true[desc_score_indices]
        distinct_value_indices = np.where(np.diff(probas_pred))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        p = tps/(tps + fps)
        p[np.isnan(p)] = 0
        r = tps/tps[-1]
        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)
        return np.r_[p[sl], 1], np.r_[r[sl], 0], thr[sl]
        # for t in threshold_idxs:
        #     tp, tn, fp, fn = self.__get_tp_tn_fp_fn(y_true, probas_pred, t)
        #     p.append(tp/(tp+fp))
        #     r.append(tp/(tp+fn))
        #     thr.append(probas_pred[t])

    def __get_tp_tn_fp_fn(self, y_true, probas_pred, thr):
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i in range(0, len(y_true)):
            if(y_true[i] == 1):
                if(i >= thr):
                    TP = TP + 1
                else:
                    FN = FN + 1
            else:
                if(probas_pred[i] >= thr):
                    FP = FP + 1
                else:
                    TN = TN + 1
        return [TP, TN, FP, FN]

    