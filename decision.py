"""Validation-only operating point and failure-aware binary classification audit.

Never use test labels to tune a threshold. These functions do not guarantee
that a classifier is useful or fair.
"""
import numpy as np
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef,
                             brier_score_loss, log_loss, confusion_matrix)


def choose_threshold(y_val, p_val, metric='balanced_accuracy', min_class_recall=0.10):
    y = np.asarray(y_val, dtype=int); p = np.asarray(p_val, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0 or not np.isfinite(p).all():
        raise ValueError('Valid finite, nonempty validation labels and probabilities required')
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError('Threshold optimization requires both validation classes')
    if metric not in ('balanced_accuracy', 'macro_f1'):
        raise ValueError('Unsupported threshold selection metric')
    # Evaluate every distinguishable partition, including thresholds near extremes.
    vals = np.unique(p)
    cuts = np.concatenate(([0.], (vals[:-1] + vals[1:]) / 2, [1.]))
    candidates = sorted(set(float(t) for t in cuts if 0 <= t <= 1) | {0.5})
    results=[]
    for threshold in candidates:
        pred=(p>=threshold).astype(int)
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        sensitivity=tp/(tp+fn) if tp+fn else 0.
        specificity=tn/(tn+fp) if tn+fp else 0.
        score=(balanced_accuracy_score(y,pred) if metric=='balanced_accuracy'
               else f1_score(y,pred,average='macro',zero_division=0))
        results.append((float(score), min(sensitivity,specificity), -abs(threshold-0.5), threshold))
    feasible=[r for r in results if r[1]>=min_class_recall]
    chosen=max(feasible if feasible else results)
    return {'threshold':float(chosen[3]),'validation_score':float(chosen[0]),
            'class_recall_constraint_feasible':bool(feasible),'objective':metric,
            'min_class_recall':float(min_class_recall),'threshold_source':'validation_only',
            'validation_n':int(len(y))}


def classification_audit(y_true, p, threshold=0.5, min_balanced_accuracy=0.55,
                         min_class_recall=0.10):
    y=np.asarray(y_true,dtype=int);p=np.asarray(p,dtype=float)
    if not len(y) or y.shape!=p.shape or not np.isfinite(p).all():
        raise ValueError('Invalid labels or predictions')
    pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    tpr=tp/(tp+fn) if tp+fn else None
    tnr=tn/(tn+fp) if tn+fp else None
    ba=float(balanced_accuracy_score(y,pred)) if len(np.unique(y))==2 else None
    collapsed=bool(len(np.unique(pred))==1)
    result={'threshold':float(threshold),'n':int(len(y)),
            'label_positive_rate':float(y.mean()),'predicted_positive_rate':float(pred.mean()),
            'collapsed_predictions':collapsed,'confusion_matrix':[[tn,fp],[fn,tp]],
            'class_0_recall':tnr,'class_1_recall':tpr,'balanced_accuracy':ba,
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'mcc':float(matthews_corrcoef(y,pred)),
            'brier':float(brier_score_loss(y,p)),
            'log_loss':float(log_loss(y,p,labels=[0,1])),
            'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'auprc':float(average_precision_score(y,p)) if len(np.unique(y))==2 else None}
    failures=[]
    if collapsed:failures.append('constant_class_predictions')
    if ba is None:failures.append('test_split_missing_class')
    elif ba<min_balanced_accuracy:failures.append('balanced_accuracy_below_floor')
    if tpr is None or tnr is None or min(tpr,tnr)<min_class_recall:
        failures.append('per_class_recall_below_floor')
    result['gate_pass']=not failures;result['failure_reasons']=failures
    return result


def prediction_probabilities(model, graphs, device='cpu'):
    import torch
    model.eval();y=[];p=[]
    with torch.no_grad():
        for g in graphs:
            g=g.to(device)
            y.append(int(g.y))
            p.append(float(torch.sigmoid(model(g)).cpu()))
    return np.asarray(y),np.asarray(p)


def node_prediction_probabilities(model, data, split, device='cpu'):
    import torch
    model.eval();d=data.to(device)
    with torch.no_grad():
        idx=getattr(d,f'{split}_idx')
        p=torch.sigmoid(model(d)[idx]).cpu().numpy()
        y=d.y[idx].cpu().numpy()
    return y,p
