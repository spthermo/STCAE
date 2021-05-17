from sklearn.metrics import roc_auc_score

def compute_auc_roc(pred, target):
    return roc_auc_score(pred, target)