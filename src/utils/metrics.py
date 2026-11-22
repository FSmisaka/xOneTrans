import numpy as np
from sklearn.metrics import roc_auc_score


def auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_pred))


def logloss_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
