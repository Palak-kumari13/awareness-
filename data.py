"""Dataset adapters and planted motif data generator."""
import random
import numpy as np
import torch
from .core import Graph,validate


def synthetic(n_graphs=160,seed=0,min_nodes=10,max_nodes=17):
    """Label 1: planted triangle, label 0: planted open 3-node path.
    Both may have accidental background triangles; motif labels are known by construction.
    Node features are constants; explanations recover planted *construction*, not a causal theorem.
    Sensitive group is synthetic, correlated with a nuisance density shift, not a protected attribute.
    """
    rng=random.Random(seed); out=[]
    for idx in range(n_graphs):
        y=idx%2; n=rng.randint(min_nodes,max_nodes); s=rng.randrange(2)
        es=set()
        # sparse background outside motif nodes, preserving exact motif construction
        for u in range(3,n):
            for v in range(u+1,n):
                if rng.random()<(.08 if s==0 else .13): es.add((u,v))
        es.update([(0,1),(1,2)])
        if y: es.add((0,2))
        # connect motif component to background consistently
        es.add((2,3))
        ed=torch.tensor(sorted(es),dtype=torch.long)
        motif=torch.tensor([u<3 and v<3 for u,v in sorted(es)],dtype=torch.bool)
        out.append(validate(Graph(torch.ones(n,1),ed,y,s,motif)))
    rng.shuffle(out)
    return out


def from_pyg(d, feature_mode='degree'):
    """Convert a single PyG graph without leaking y into feature construction."""
    if d.y is None or d.y.numel()!=1:
        raise ValueError('Graph-level binary labels required (one y per graph)')
    y=int(d.y.item()); assert y in (0,1)
    n=int(d.num_nodes)
    es=set()
    for u,v in d.edge_index.T.cpu().tolist():
        if u!=v: es.add((min(u,v),max(u,v)))
    edges=torch.tensor(sorted(es),dtype=torch.long).reshape(-1,2)
    if d.x is not None:
        x=d.x.float().cpu()
    elif feature_mode=='degree':
        deg=torch.zeros(n,1)
        for u,v in es:
            deg[u]+=1;deg[v]+=1
        x=deg/(max(n-1,1))
    elif feature_mode=='ones': x=torch.ones(n,1)
    else: raise ValueError('Unsupported feature mode')
    return validate(Graph(x,edges,y))


def load_tu(name,root='data'):
    try: from torch_geometric.datasets import TUDataset
    except ImportError as e: raise ImportError('Install torch-geometric for TU datasets') from e
    ds=TUDataset(root=f'{root}/{name}',name=name)
    graphs=[from_pyg(d) for d in ds]
    n_labels=set(g.y for g in graphs)
    if n_labels!={0,1}: raise ValueError(f'Expected binary labels, got {n_labels}')
    return graphs


def split(graphs,seed=0):
    from sklearn.model_selection import train_test_split
    ids=np.arange(len(graphs)); y=np.array([g.y for g in graphs]);
    tr,te=train_test_split(ids,test_size=.2,stratify=y,random_state=seed)
    tr,va=train_test_split(tr,test_size=.25,stratify=y[tr],random_state=seed)
    return [[graphs[int(i)] for i in part] for part in (tr,va,te)]
