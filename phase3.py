"""Phase 3: higher-order intervention effects, concept tests, motif gates and leakage controls."""
import itertools
import math
import numpy as np
import torch
from .core import intervention, Graph
from .audit import prob,rank_edges,audit_graph
from .trainlib import train,metrics,seed_all
from .phase2 import FullGraphBackbone

@torch.no_grad()
def group_removal_audit(model,g,method='deletion',group_size=2,top_n=8,random_reps=20,seed=0):
    """Assess joint edge removals; interaction = joint effect minus individual effects.
    Computes signed decrease in confidence of the originally predicted class.
    """
    if group_size<2:raise ValueError('group_size must be >=2')
    original_selector=getattr(model,'selector_on',None)
    if original_selector is not None:model.selector_on=False
    try:
        full=prob(model,g); sign=1 if full>=.5 else -1
        m=len(g.edges)
        scores=rank_edges(model,g,method)
        top=np.argsort(-scores,kind='stable')[:min(top_n,m)].tolist()
        singleton={i:sign*(full-prob(model,intervention(g,[i]))) for i in top}
        combos=list(itertools.combinations(top,group_size))
        rng=np.random.default_rng(seed)
        if len(combos)>1000:
            chosen=rng.choice(len(combos),1000,replace=False)
            combos=[combos[int(j)] for j in chosen]
        results=[]
        for idx in combos:
            delta=sign*(full-prob(model,intervention(g,list(idx))))
            results.append({'removed_edges':list(map(int,idx)),'effect':float(delta),
                            'interaction':float(delta-sum(singleton[j] for j in idx))})
        controls=[]
        for _ in range(random_reps):
            idx=rng.choice(m,size=min(group_size,m),replace=False).tolist() if m else []
            controls.append(sign*(full-prob(model,intervention(g,idx))))
        return {'full_prob':full,'groups':results,'random_effect_mean':float(np.mean(controls)) if controls else None,
                'random_effect_std':float(np.std(controls)) if controls else None}
    finally:
        if original_selector is not None:model.selector_on=original_selector

@torch.no_grad()
def concept_interventions(model,g,scales=(0.,.5,1.,1.5)):
    """Controlled change of explicit concept input to classifier at fixed embedding.
    Only applies to vgr.core.GNN and measures *model response*, not intervention realism.
    """
    if not getattr(model,'concepts_on',False):raise ValueError('Model has no explicit concept channels')
    details=model(g,return_details=True)
    pooled=details['representation'][:-3].detach()
    base=details['concepts'].detach()
    out=[]
    for j in range(3):
        curve=[]
        for scale in scales:
            c=base.clone();c[j]=base[j]*float(scale)
            p=torch.sigmoid(model.out(torch.cat([pooled,c])).squeeze()).item()
            curve.append({'scale':float(scale),'prob':p})
        out.append({'concept_index':j,'base_value':float(base[j]),'curve':curve})
    return out

def motif_gate(model,graphs,method='selector',budget=.5,seed=0):
    """Ground truth is generator-planted edges; comparison to equal-budget random baseline."""
    if not graphs or any(g.motif is None for g in graphs):raise ValueError('Synthetic graphs with motif required')
    rows=[audit_graph(model,g,method=method,fraction=budget,seed=seed+i) for i,g in enumerate(graphs)]
    expected=[min(len(g.edges),max(1,round(len(g.edges)*budget)))/len(g.edges) for g in graphs]
    precision=np.mean([r['motif_precision'] for r in rows])
    return {'precision':float(precision),'recall':float(np.mean([r['motif_recall'] for r in rows])),
            'random_expected_precision':float(np.mean([g.motif.float().mean().item() for g in graphs])),
            'pass_random_precision':bool(precision>np.mean([g.motif.float().mean().item() for g in graphs])),
            'research_success':False,
            'research_success_reason':'Requires paired multi-seed uncertainty and predictive utility gates; point estimates are insufficient'}

def shuffle_label_audit(trainset,valset,testset,seed=0,epochs=30,hidden=32,device='cpu'):
    """Permutation-label retraining sanity control. Test original labels remain unchanged.
    Outcomes of a single shuffle are not statistical significance evidence.
    """
    rng=np.random.default_rng(seed)
    ys=[g.y for g in trainset];shuffled=rng.permutation(ys)
    def replace_labels(gs,values):
        return [Graph(g.x,g.edges,int(y),g.sensitive,g.motif) for g,y in zip(gs,values)]
    shuffled_train=replace_labels(trainset,shuffled)
    val_labels=rng.permutation([g.y for g in valset]);shuffled_val=replace_labels(valset,val_labels)
    seed_all(seed)
    model=FullGraphBackbone(trainset[0].x.size(1),hidden=hidden,kind='gin')
    model,_=train(model,shuffled_train,shuffled_val,epochs=epochs,seed=seed,device=device)
    return metrics(model,testset,device)
