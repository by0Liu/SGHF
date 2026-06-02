from sklearn import metrics
from sklearn.metrics import roc_auc_score
import math


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(14):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


def calculate_auc(y_pred, y_gt, num_classes):
    '''calculate the mean AUC'''
    auc_each_class = []
    nan_index = []

    mean_auc = 0

    for index in range(num_classes):

        pred = y_pred[:, index]
        label = y_gt[:, index]

        fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=1)

        auc = metrics.auc(fpr, tpr)

        if(math.isnan(auc)):
            nan_index.append(index)
            auc = 0.0

        auc_each_class.append(auc)
        mean_auc += auc

    mean_auc = float(mean_auc) / float(num_classes)

    return mean_auc, auc_each_class




