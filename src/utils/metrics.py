import torch
from sklearn.metrics import roc_auc_score


def compute_auc(y_true, y_pred):
    """
    y_true, y_pred: tensors of shape [N, C]
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    aucs = []
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = float("nan")
        aucs.append(auc)

    return aucs
