"""Validation-only non-regression selection for binary selected-node models.
Never infer success from a constant classifier or from test-selected hyperparameters.
"""
import copy
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from .integrated import train_selected_node
from .fairness import fairness_gaps

@torch.no_grad()
def predict(model,data,split):
    model.eval()
    idx=getattr(data,split+'_idx')
    p=model(data)[idx].sigmoid().cpu().numpy()
    return data.y[idx].cpu().numpy(), data.sensitive[idx].cpu().numpy(),p

def evaluate(y,s,p,threshold=.5):
    pred=(p>=threshold).astype(int)
    out={'balanced_accuracy':float(balanced_accuracy_score(y,pred)) if len(np.unique(y))==2 else None,
         'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
         'positive_rate':float(np.mean(pred)), 'collapsed':bool(len(np.unique(pred))<2)}
    out.update(fairness_gaps(y,p,s,threshold))
    return out

def select_validation_threshold(y,p):
    """Maximize balanced accuracy on validation only. Deterministic tie preference near .5."""
    if len(np.unique(y))<2: return .5
    candidates=np.unique(np.r_[0.5,np.nextafter(np.unique(p),float('inf'))])
    scores=[balanced_accuracy_score(y,p>=t) for t in candidates]
    best=max(scores)
    return float(min((float(t) for t,z in zip(candidates,scores) if z>=best-1e-12),key=lambda t:(abs(t-.5),t)))

def select_nonregressing(data,configs,seed=0,epochs=80,utility_tolerance=.02,metric='dp_gap'):
    """Train baseline and candidates with identical seed, splits and epochs.
    Select fairness penalty only if validation BA and AUROC are non-inferior to baseline,
    classification is nonconstant, and gap is strictly smaller; otherwise use baseline.
    Test labels/predictions are accessed ONLY after selection.
    """
    if not configs or configs[0].get('fair_weight',None)!=0.:
        raise ValueError('First configuration must be the unregularized baseline')
    models=[];validation=[]
    for cfg in configs:
        model,hist=train_selected_node(data,seed=seed,epochs=epochs,**cfg)
        y,s,p=predict(model,data,'val')
        t=select_validation_threshold(y,p)
        m=evaluate(y,s,p,t)
        models.append(model);validation.append({'config':cfg,'threshold':t,'metrics':m})
    ref=validation[0]['metrics']
    def acceptable(m):
        if m['collapsed'] or m['balanced_accuracy'] is None or m['auroc'] is None:return False
        if ref['balanced_accuracy'] is not None and m['balanced_accuracy']+utility_tolerance<ref['balanced_accuracy']:return False
        if ref['auroc'] is not None and m['auroc']+utility_tolerance<ref['auroc']:return False
        return True
    chosen=0
    if acceptable(ref) and ref.get(metric) is not None:
        for j,entry in enumerate(validation[1:],1):
            m=entry['metrics']
            if acceptable(m) and m.get(metric) is not None and m[metric]<validation[chosen]['metrics'][metric]-1e-10:
                chosen=j
    y,s,p=predict(models[chosen],data,'test')
    chosen_test=evaluate(y,s,p,validation[chosen]['threshold'])
    y0,s0,p0=predict(models[0],data,'test')
    baseline_test=evaluate(y0,s0,p0,validation[0]['threshold'])
    return {'seed':seed,'selected_index':chosen,'used_baseline_fallback':chosen==0,
            'validation_candidates':validation,'selected_test':chosen_test,
            'baseline_test':baseline_test,
            'test_nonregression_observed':bool(not chosen_test['collapsed'] and
                chosen_test['balanced_accuracy'] is not None and baseline_test['balanced_accuracy'] is not None and
                chosen_test['balanced_accuracy']+utility_tolerance>=baseline_test['balanced_accuracy'])}
