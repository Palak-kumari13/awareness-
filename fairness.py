"""Phase 5: observable node-level fairness optimization with utility floor.
DP/EO gaps are descriptive; they cannot determine whether disparity is unjustified.
"""
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import (roc_auc_score,average_precision_score,accuracy_score,
                             f1_score,balanced_accuracy_score,matthews_corrcoef,brier_score_loss)
from .node import NodeGNN,validate_node
from .trainlib import seed_all
from .decision import choose_threshold,classification_audit,node_prediction_probabilities

def fairness_gaps(y,p,s,threshold=.5):
    """Group gaps calculated only on labeled groups; undefined strata -> None."""
    y=np.asarray(y);p=np.asarray(p);s=np.asarray(s)
    pred=(p>=threshold).astype(int)
    valid=np.isin(s,[0,1]);ans={'groups_known':int(valid.sum()),'n':len(s)}
    for g in (0,1):
        mask=valid&(s==g)
        ans[f'group_{g}_n']=int(mask.sum())
        ans[f'group_{g}_accuracy']=float((pred[mask]==y[mask]).mean()) if mask.any() else None
        ans[f'group_{g}_positive_rate']=float(pred[mask].mean()) if mask.any() else None
        ans[f'group_{g}_tpr']=float(pred[mask&(y==1)].mean()) if (mask&(y==1)).any() else None
        ans[f'group_{g}_fpr']=float(pred[mask&(y==0)].mean()) if (mask&(y==0)).any() else None
    def gap(key):
        a,b=ans[f'group_0_{key}'],ans[f'group_1_{key}']
        return abs(a-b) if a is not None and b is not None else None
    ans['dp_gap']=gap('positive_rate');ans['eo_gap']=gap('tpr')
    t=gap('tpr');f=gap('fpr')
    ans['equalized_odds_gap']=max(t,f) if t is not None and f is not None else None
    return ans

def node_metrics(model,data,split='test',device='cpu',threshold=.5):
    model.eval();d=data.to(device)
    idx=getattr(d,f'{split}_idx')
    with torch.no_grad(): p=torch.sigmoid(model(d)[idx]).cpu().numpy()
    y=d.y[idx].cpu().numpy();s=d.sensitive[idx].cpu().numpy();pred=(p>=threshold).astype(int)
    res={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
         'f1':float(f1_score(y,pred,zero_division=0)),
         'balanced_accuracy':float(balanced_accuracy_score(y,pred)) if len(np.unique(y))==2 else None,
         'mcc':float(matthews_corrcoef(y,pred)),
         'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
         'auprc':float(average_precision_score(y,p)) if len(np.unique(y))==2 else None,
         'brier':float(brier_score_loss(y,p)),
         'predicted_positive_rate':float(pred.mean()), 'collapsed':bool(len(set(pred))==1),
         'threshold':float(threshold)}
    res.update(fairness_gaps(y,p,s,threshold));return res

def dp_surrogate(prob,s):
    a=prob[s==0];b=prob[s==1]
    return (a.mean()-b.mean()).abs() if len(a) and len(b) else prob.new_zeros(())

def eo_surrogate(prob,y,s):
    a=prob[(s==0)&(y==1)];b=prob[(s==1)&(y==1)]
    return (a.mean()-b.mean()).abs() if len(a) and len(b) else prob.new_zeros(())

def train_node(data,epochs=100,hidden=32,kind='sage',fair_weight=0.,fair_kind='dp',
               utility_floor=None,seed=0,lr=.005,device='cpu'):
    """Full-batch transductive node training. Validation selection constrained by
    accuracy >= utility_floor, if supplied; infeasible returns best unconstrained
    checkpoint and a false feasible flag (never silently claims success).
    """
    validate_node(data)
    if fair_weight<0:raise ValueError('fair_weight must be >=0')
    if fair_kind not in ('dp','eo'):raise ValueError(fair_kind)
    if fair_weight and not all((data.sensitive[data.train_idx]>=0).tolist()):
        raise ValueError('Fairness training requires sensitive group for all training nodes')
    seed_all(seed);d=data.to(device)
    model=NodeGNN(d.x.size(1),hidden=hidden,kind=kind).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    best=None;best_loss=float('inf');best_feasible=None;best_feasible_loss=float('inf');history=[]
    for epoch in range(epochs):
        model.train();opt.zero_grad();logits=model(d);idx=d.train_idx
        loss=F.binary_cross_entropy_with_logits(logits[idx],d.y[idx].float())
        if fair_weight:
            p=logits[idx].sigmoid();s=d.sensitive[idx];y=d.y[idx]
            loss=loss+fair_weight*(dp_surrogate(p,s) if fair_kind=='dp' else eo_surrogate(p,y,s))
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        model.eval()
        with torch.no_grad():
            v=model(d)[d.val_idx]
            vl=F.binary_cross_entropy_with_logits(v,d.y[d.val_idx].float()).item()
            acc=((v.sigmoid()>=.5)==d.y[d.val_idx]).float().mean().item()
        if vl<best_loss:best_loss=vl;best=copy.deepcopy(model.state_dict())
        feasible=utility_floor is None or acc>=utility_floor
        if feasible and vl<best_feasible_loss:
            best_feasible_loss=vl;best_feasible=copy.deepcopy(model.state_dict())
        history.append({'epoch':epoch,'val_bce':vl,'val_accuracy':acc,'feasible':feasible})
    passed=best_feasible is not None
    model.load_state_dict(best_feasible if passed else best)
    return model,history,{'utility_floor':utility_floor,'floor_feasible':passed,
                           'selected_val_bce':best_feasible_loss if passed else best_loss}

def fairness_sweep(data,weights=(0.,.01,.1,.5),utility_drop=.02,epochs=100,seed=0,**kwargs):
    """Baseline-matched validation utility floor; test split never selects weight.
    Returns all runs for Pareto trade-off reporting, not claims of fairness improvement.
    """
    models={};runs=[]
    base,_,status=train_node(data,epochs=epochs,seed=seed,fair_weight=0.,**kwargs)
    base_val=node_metrics(base,data,'val');floor=max(0.,base_val['accuracy']-utility_drop)
    for w in weights:
        if w==0: model=base;state={'floor_feasible':base_val['accuracy']>=floor,'utility_floor':floor}
        else:model,_,state=train_node(data,epochs=epochs,seed=seed,fair_weight=float(w),utility_floor=floor,**kwargs)
        yv,pv=node_prediction_probabilities(model,data,'val')
        yt,pt=node_prediction_probabilities(model,data,'test')
        threshold_info=choose_threshold(yv,pv)
        test_audit=classification_audit(yt,pt,threshold_info['threshold'])
        runs.append({'fair_weight':float(w),'status':state,
                     'validation':node_metrics(model,data,'val'),
                     'test':node_metrics(model,data,'test'),
                     'decision_audit':{'validation_selection':threshold_info,
                                       'test_at_validation_threshold':test_audit},
                     'fairness_success_claim_allowed':bool(state['floor_feasible'] and test_audit['gate_pass'])})
        models[float(w)]=model
    return runs,models
