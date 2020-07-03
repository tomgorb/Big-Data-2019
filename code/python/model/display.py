import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score

logger = logging.getLogger(__name__)


def f1_score_threshold(y_predictions, y_true, precision_loop=4):

    for i in range(precision_loop):
        preds = []
        scores = []
        index_out = np.asarray([False] * 10)
        espilon = 1e-2
        if i == 0:
            thresholds = np.linspace(0+espilon, 1-espilon, 10)
        else:
            bound_inf = threshold - (1/(np.power(10, i)))
            bound_sup = threshold + (1/(np.power(10, i)))
            if bound_inf < 0:
                bound_inf = 0
            if bound_sup > 1:
                bound_sup = 1
            thresholds = np.linspace(bound_inf, bound_sup, 10)
        for idx, threshold in enumerate(thresholds):
            y_pred = []
            for row in y_predictions:
                if row >= threshold:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            scores.append(f1_score(y_true, y_pred))
            preds.append(y_pred)
        scores = np.asarray(scores)[np.where(~index_out)]
        thresholds = np.asarray(thresholds)[np.where(~index_out)]
        score = np.max(scores)
        threshold = thresholds[np.argmax(scores)]
        y_pred = preds[np.argmax(scores)]
        y_pred = np.asarray(y_pred)
    return score, threshold, y_pred


def get_fpr(true_labels, pred_labels):
 
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    return FP/float(FP+TN)


def get_precision(true_labels, pred_labels):
 
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    return TP/float(TP+FP)


def get_tpr(true_labels, pred_labels):
 
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    return TP/float(TP+FN)


def get_roc_curve(y_test, y_pred):

    nbOnes = y_test.sum()
    nbZeros = y_test.shape[0] - nbOnes
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, thres = precision_recall_curve(y_test, y_pred)
    b_f1_s, b_f1_t, b_f1_p = f1_score_threshold(y_pred, y_test, precision_loop=4)
    best_f1_score, best_f1_threshold, best_f1_y_pred = b_f1_s, b_f1_t, b_f1_p
    best_f1_precision = get_precision(y_test, best_f1_y_pred)
    best_f1_tpr = get_tpr(y_test, best_f1_y_pred)
    best_f1_fpr = get_fpr(y_test, best_f1_y_pred)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(fpr, tpr, label='ROC curve for LSTM with Keras (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('[0]: {0} / [1]: {1}'.format(nbZeros, nbOnes))
    plt.legend(loc="lower right")
    return fig
