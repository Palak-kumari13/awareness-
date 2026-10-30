"""Phase 2: true full-graph baseline comparisons and graph-signal controls.
All models use the same train/validation/test graph IDs and fixed seed.
"""
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .core import GNN, Graph
from .trainlib import train, metrics, seed_all

class FullGraphBackbone(nn.Module):
    """Dense small-graph implementations of GIN, GCN, GraphSAGE and feature-only MLP.
    This class is intentionally limited to small graphs due to NxN adjacency.
    """
    def __init__(self,input_dim,hidden=32,layers=2,kind='gin'):
        super().__init__()
        if kind not in ('gin','gcn','sage','mlp'): raise ValueError(kind)
        self.kind=kind
        self.proj=nn.Linear(input_dim,hidden)
        self.self_layers=nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(layers)])
        self.neigh_layers=nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in range(layers)])
        self.out=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,g):
        h=F.relu(self.proj(g.x))
        n=len(g.x)
        a=h.new_zeros(n,n)
        if self.kind!='mlp' and len(g.edges):
            u,v=g.edges.T
            a[u,v]=1.;a[v,u]=1.
        for self_linear,neigh_linear in zip(self.self_layers,self.neigh_layers):
            if self.kind=='gin':
                h=F.relu(self_linear(h+a@h))
            elif self.kind=='gcn':
                at=a+torch.eye(n,device=h.device)
                d=at.sum(-1).clamp(min=1).pow(-.5)
                h=F.relu(self_linear((d[:,None]*at*d[None,:])@h))
            elif self.kind=='sage':
                nbr=(a@h)/a.sum(-1,keepdim=True).clamp(min=1)
                h=F.relu(self_linear(h)+neigh_linear(nbr))
            else:
                h=F.relu(self_linear(h))
        return self.out(h.mean(0)).squeeze()

def degree_preserving_rewire(g,seed=0,swaps_per_edge=5):
    """Edge swaps preserve every node degree, undirected simplicity and all features.
    Connectivity is not guaranteed. Does not preserve labels on arbitrary graphs.
    """
    rng=random.Random(seed)
    es=set(map(tuple,g.edges.detach().cpu().tolist()))
    if len(es)<2: return g
    edge_list=sorted(es)
    for _ in range(swaps_per_edge*len(es)):
        (a,b),(c,d)=rng.sample(edge_list,2)
        if len({a,b,c,d})<4:continue
        # try either cross matching; never introduce self loops or parallel edges
        choice=rng.choice([((a,c),(b,d)),((a,d),(b,c))])
        f,h=(tuple(sorted(e)) for e in choice)
        if f==h or f in es or h in es:continue
        es.remove((a,b));es.remove((c,d));es.add(f);es.add(h)
        edge_list=sorted(es)
    new_edges=torch.as_tensor(sorted(es),dtype=torch.long,device=g.edges.device).reshape(-1,2)
    return Graph(g.x,new_edges,g.y,g.sensitive,g.motif)

def majority_result(trainset,testset):
    positive=sum(g.y for g in trainset)>=len(trainset)/2
    yy=np.array([g.y for g in testset]);pred=np.full(len(testset),int(positive))
    from sklearn.metrics import f1_score,balanced_accuracy_score,matthews_corrcoef
    return {'accuracy':float(np.mean(pred==yy)), 'f1':float(f1_score(yy,pred,zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(yy,pred)),
            'mcc':float(matthews_corrcoef(yy,pred)), 'predicted_positive_rate':float(pred.mean()),
            'auroc':.5 if len(set(yy))==2 else float('nan')}

def compare_baselines(trainset,valset,testset,epochs=30,hidden=32,seed=0,device='cpu'):
    """Train controls with independent initializations and same evaluation split."""
    results={'majority':majority_result(trainset,testset)}
    for kind in ('mlp','gin','gcn','sage'):
        seed_all(seed)
        model=FullGraphBackbone(trainset[0].x.size(1),hidden=hidden,kind=kind)
        model,_=train(model,trainset,valset,epochs=epochs,device=device,seed=seed)
        results[kind]=metrics(model,testset,device)
        if kind=='gin':
            # Evaluate rewired topology without retraining; this is an intervention control.
            rewired=[degree_preserving_rewire(g,seed=seed*100000+i) for i,g in enumerate(testset)]
            results['gin_evaluate_rewired']=metrics(model,rewired,device)
            # Train a separate model on rewired training/validation graphs.
            rw_tr=[degree_preserving_rewire(g,seed=seed*100000+i) for i,g in enumerate(trainset)]
            rw_va=[degree_preserving_rewire(g,seed=seed*200000+i) for i,g in enumerate(valset)]
            seed_all(seed)
            rw_model=FullGraphBackbone(trainset[0].x.size(1),hidden=hidden,kind='gin')
            rw_model,_=train(rw_model,rw_tr,rw_va,epochs=epochs,device=device,seed=seed)
            results['gin_train_rewired']=metrics(rw_model,rewired,device)
    return results
