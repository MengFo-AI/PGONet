import os 
import torch
import numpy as np 
import scipy


def Matthews(y_true, y_pred, eps:float=1e-10):
    y_pred_pos = torch.round(torch.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    #y_pos = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    y_pos = torch.round(torch.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = torch.sum(y_pos * y_pred_pos)
    tn = torch.sum(y_neg * y_pred_neg)

    fp = torch.sum(y_neg * y_pred_pos)
    fn = torch.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + eps)





def f1_torch(y_true, y_pred, eps:float=1e-10):
    def recall(y_true, y_pred, eps:float=eps):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
        possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + eps)#epsilon())
        return recall

    def precision(y_true, y_pred, eps:float=eps):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
        predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + eps)#epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+ eps))