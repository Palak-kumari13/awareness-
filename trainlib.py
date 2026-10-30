import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,balanced_accuracy_score,matthews_corrcoef,brier_score_loss
from .core import GNN,Graph


def seed_all(seed):
    random.seed(seed); np.random.seed(seed);torch.manual_seed(seed)

def perturb(g, rate=.03, seed=0):
    """Undirected edge flips; not assumed label-preserving on real datasets."""
    rng=random.Random(seed)
    n=len(g.x); edges=set(map(tuple,g.edges.cpu().tolist())); count=max(1,round(rate*len(edges)))
    for _ in range(count):
        u,v=sorted(rng.sample(range(n),2)); e=(u,v)
        if e in edges:edges.remove(e)
        else:edges.add(e)
    edge_tensor=torch.tensor(sorted(edges),dtype=torch.long,device=g.x.device).reshape(-1,2)
    return Graph(g.x,edge_tensor,g.y,g.sensitive)


def metrics(model,graphs,device='cpu'):
    model.eval();y=[];p=[];group=[]
    with torch.no_grad():
        for g in graphs:
            g=g.to(device); y.append(g.y);p.append(float(torch.sigmoid(model(g)).cpu()));group.append(g.sensitive)
    yp=[int(v>=.5) for v in p]; res={
        'auroc':roc_auc_score(y,p) if len(set(y))==2 else float('nan'),
        'auprc':average_precision_score(y,p) if len(set(y))==2 else float('nan'),
        'f1':f1_score(y,yp,zero_division=0),
        'balanced_accuracy':balanced_accuracy_score(y,yp),
        'mcc':matthews_corrcoef(y,yp),
        'brier':brier_score_loss(y,p),
        'predicted_positive_rate':float(np.mean(yp)),
        'accuracy':float(np.mean(np.array(y)==np.array(yp))) }
    if set(group)=={0,1}:
        p=np.array(p);y=np.array(y); yp=np.array(yp);s=np.array(group)
        res['dp_gap']=float(abs(yp[s==0].mean()-yp[s==1].mean()))
        positives=[(s==a)&(y==1) for a in (0,1)]
        res['eo_gap']=float(abs(yp[positives[0]].mean()-yp[positives[1]].mean())) if all(z.any() for z in positives) else float('nan')
    return res


def train(model,trainset,valset,epochs=30,lr=.002,stage='V1',device='cpu',seed=0,
          fair_weight=0.,robust_weight=0.,reference=None,selection='balanced',
          class_weighting=True,patience=25):
    """Train with TRAIN-only class weighting and VALIDATION-only checkpoint decisions.
    Balanced checkpoint selection prioritizes ability to rank both validation classes
    (AUROC), using validation BCE to break ties. Does not touch test labels.
    The returned model is NOT guaranteed to pass the decision gate.
    """
    from sklearn.metrics import roc_auc_score
    if selection not in ('balanced','bce'):raise ValueError(selection)
    if not trainset or not valset:raise ValueError('Empty training/validation split')
    seed_all(seed);model.to(device);opt=torch.optim.Adam(model.parameters(),lr=lr)
    npos=sum(int(g.y) for g in trainset);nneg=len(trainset)-npos
    if not npos or not nneg:raise ValueError('Both train labels required')
    pos_weight=torch.tensor([min(10.,nneg/npos) if class_weighting else 1.],device=device)
    best=None;best_score=None;history=[];stale=0
    for epoch in range(epochs):
        model.train();rng=random.Random(seed+epoch)
        order=list(range(len(trainset)));rng.shuffle(order)
        for start in range(0,len(order),16):
            chunk=[trainset[i].to(device) for i in order[start:start+16]]
            logits=torch.stack([model(g) for g in chunk]);ys=torch.tensor([g.y for g in chunk],dtype=torch.float,device=device)
            loss=F.binary_cross_entropy_with_logits(logits,ys,pos_weight=pos_weight)
            if fair_weight:
                groups=[g.sensitive for g in chunk]
                if 0 in groups and 1 in groups:
                    s=torch.tensor(groups,device=device);pp=logits.sigmoid()
                    loss=loss+fair_weight*(pp[s==0].mean()-pp[s==1].mean()).abs()
            if robust_weight:
                edits=[perturb(g,.03,seed+epoch*10000+idx) for idx,g in enumerate(chunk)]
                edit_logits=torch.stack([model(g) for g in edits])
                loss=loss+robust_weight*F.mse_loss(logits.sigmoid(),edit_logits.sigmoid())
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        model.eval()
        with torch.no_grad():
            logits_val=torch.stack([model(g.to(device)) for g in valset])
            labels_val=torch.tensor([g.y for g in valset],dtype=torch.float,device=device)
            vloss=float(F.binary_cross_entropy_with_logits(logits_val,labels_val).item())
            probs=logits_val.sigmoid().cpu().numpy();labels=labels_val.cpu().numpy()
        val_auc=float(roc_auc_score(labels,probs)) if len(set(labels))==2 else float('nan')
        # Use validation ranking before thresholding; minimum BCE breaks AUROC ties.
        score=(-val_auc,vloss) if selection=='balanced' and np.isfinite(val_auc) else (vloss,)
        history.append({'epoch':epoch,'validation_bce':vloss,'validation_auroc':val_auc,
                        'selection':selection})
        if best_score is None or score<best_score:
            best_score=score;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if patience and stale>=patience:break
    model.load_state_dict(best)
    return model,history
