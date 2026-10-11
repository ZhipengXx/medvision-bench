import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)
from .data import CLASSES


def metrics(y, probabilities):
    y, p = np.asarray(y), np.asarray(probabilities)
    if p.shape != (len(y), 7) or not np.isfinite(p).all():
        raise ValueError('Invalid seven-class probabilities')
    pred = p.argmax(1)
    result = {'accuracy': float(accuracy_score(y, pred)), 'n': len(y)}
    labels = np.arange(7)
    for avg in ['macro', 'weighted']:
        a, b, c, _ = precision_recall_fscore_support(y, pred, labels=labels, average=avg, zero_division=0)
        result.update({f'{avg}_precision': float(a), f'{avg}_recall': float(b), f'{avg}_f1': float(c)})
    a, b, c, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    rows, aucs = [], []
    for k, name in enumerate(CLASSES):
        binary = y == k
        auc = float(roc_auc_score(binary, p[:, k])) if 0 < binary.sum() < len(y) else None
        rows.append(dict(class_name=name, support=int(s[k]), precision=float(a[k]), recall=float(b[k]),
                         f1=float(c[k]), auroc=auc,
                         auroc_reason=None if auc is not None else 'No positive or no negative sample'))
        if auc is not None:
            aucs.append(auc)
    result['macro_auroc'] = float(np.mean(aucs)) if aucs else None
    result['auroc_valid_classes'] = [r['class_name'] for r in rows if r['auroc'] is not None]
    cm = confusion_matrix(y, pred, labels=labels)
    result['confusion_counts'] = cm.tolist()
    result['confusion_row_normalized'] = (cm / np.maximum(cm.sum(1, keepdims=True), 1)).tolist()
    return result, rows
