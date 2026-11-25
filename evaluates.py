import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc,roc_auc_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy as np
import pandas as pd
from decimal import Decimal
import math
import torch
from sklearn import metrics


def find_best_cutoff(y_true,y_proba):
    fpr, tpr, thresholds = roc_curve(y_true,y_proba)
    best_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_index]
    best_point = [fpr[best_index],tpr[best_index]]
    return best_threshold,best_point,fpr,tpr

def roc_plot(best_threshold,best_point,fpr,tpr):
    plt.figure(figsize=(7,5.5),dpi=80,facecolor=None,frameon=True)
    plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC curve')
    plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
    plt.plot(best_point[0],best_point[1],marker='o',color='r',markersize=2)
    plt.xlim([-0.0045,1.0045])
    plt.ylim([-0.0045,1.0045])
    plt.text(best_point[0],best_point[1],'Threshold:%0.2f' % best_threshold)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()

def metrics_evaluate(y_true, y_proba, best_threshold):
    y_score = [1 if i > best_threshold else 0 for i in y_proba]
    TN,FP,FN,TP = confusion_matrix(y_true,y_score).ravel()
    roc_auc = roc_auc_score(y_true,y_proba)
    accuracy = (TP+TN)/float(TP+TN+FP+FN)
    recall = TP/float(TP+FN)
    specificity = TN/float(TN+FP)
    precision = TP/float(TP+FP)
    F1_score=(2*precision*recall)/(precision+recall)
    roc_auc,accuracy,recall,specificity,F1_score = map(lambda i:Decimal(i).quantize(Decimal('0.0001'),rounding='ROUND_HALF_UP'),
                                                                 [roc_auc,accuracy,recall,specificity,F1_score])
    return roc_auc,accuracy,recall,specificity,F1_score



def metrics_performance(y_true, y_pred, y_proba):
    TN,FP,FN,TP = confusion_matrix(y_true,y_pred).ravel()
    roc_auc = roc_auc_score(y_true,y_proba)
    accuracy = (TP+TN)/float(TP+TN+FP+FN)
    recall = TP/float(TP+FN)
    specificity = TN/float(TN+FP)
    precision = TP/float(TP+FP)
    F1_score=(2*precision*recall)/(precision+recall)
    roc_auc,accuracy,recall,specificity,F1_score = map(lambda i:Decimal(i).quantize(Decimal('0.0001'),rounding='ROUND_HALF_UP'),
                                                                 [roc_auc,accuracy,recall,specificity,F1_score])
    return roc_auc,accuracy,recall,specificity,F1_score


def our_confidence(Y, alpha, uncertainty_type='max_prob', lamb1=1.0, lamb2=1.0):
    corrects = (Y.squeeze() == alpha.max(-1)[1]).cpu().detach().numpy()

    if uncertainty_type == 'max_alpha':
        scores = alpha.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'max_prob':
        p = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        scores = p.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'max_modified_prob':
        num_classes = alpha.shape[-1]
        evidence = alpha - lamb2
        S = evidence + lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence) + lamb2 * num_classes
        p = alpha / S
        scores = p.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == 'alpha0':
        scores = alpha.sum(-1).cpu().detach().numpy()
    elif uncertainty_type == 'differential_entropy':
        eps = 1e-6
        alpha = alpha + eps
        alpha0 = alpha.sum(-1)
        log_term = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha0)
        # digamma_term = torch.sum((alpha - lamb2) * (torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))),dim=-1)
        digamma_term = torch.sum((alpha - 1) * (torch.digamma(alpha) - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))),dim=-1)
        differential_entropy = log_term - digamma_term
        scores = - differential_entropy.cpu().detach().numpy()
    elif uncertainty_type == 'mutual_information':
        eps = 1e-6
        alpha = alpha + eps
        alpha0 = alpha.sum(-1)
        probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=-1)
        digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(
            alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) + 1.0)
        dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        exp_data_uncertainty = -1 * torch.sum(dirichlet_mean * digamma_term, dim=-1)
        distributional_uncertainty = total_uncertainty - exp_data_uncertainty
        scores = - distributional_uncertainty.cpu().detach().numpy()
    else:
        raise ValueError(f"Invalid uncertainty type: {uncertainty_type}!")

    if uncertainty_type in ['differential_entropy', 'mutual_information']:
        unc = -scores
    else:
        unc = scores
    scores_norm = (unc - min(unc)) / (max(unc) - min(unc))
    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)
    return auroc, aupr, scores,scores_norm






