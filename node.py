"""Phase 4: node-level binary classification on one attributed graph.
Node IDs are stable across the graph, splits, explanations and evaluations.
NPZ schema is explicit; raw Bail/Credit/German/Pokec source formats require verified adapters.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from .core import Graph,validate

@dataclass
class NodeData:
    x: torch.Tensor          # [N,F]
    edges: torch.Tensor      # [M,2] unique u<v
    y: torch.Tensor          # [N] binary labels
    sensitive: torch.Tensor  # [N] values 0/1, or -1 unknown
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    def to(self,device):
        return NodeData(*(getattr(self,k).to(device) for k in self.__dataclass_fields__))

def validate_node(data):
    n=data.x.size(0)
    assert data.x.ndim==2 and data.edges.ndim==2 and data.edges.shape[1]==2
    assert data.y.shape==(n,) and data.sensitive.shape==(n,)
    assert set(torch.unique(data.y).tolist())<= {0,1}
    assert set(torch.unique(data.sensitive).tolist())<= {-1,0,1}
    if len(data.edges):
        assert ((data.edges[:,0]>=0)&(data.edges[:,0]<data.edges[:,1])&(data.edges[:,1]<n)).all()
        assert torch.unique(data.edges,dim=0).shape[0]==len(data.edges)
    ids=[]
    for k in ('train_idx','val_idx','test_idx'):
        z=getattr(data,k)
        assert z.dtype==torch.long and ((z>=0)&(z<n)).all()
        ids+=z.cpu().tolist()
    assert len(set(ids))==len(ids) and len(ids)>0
    return data

def node_splits(y,seed=0,train_fraction=.6,val_fraction=.2):
    yy=np.asarray(y);ids=np.arange(len(yy))
    tr,hold=train_test_split(ids,train_size=train_fraction,stratify=yy,random_state=seed)
    va,te=train_test_split(hold,train_size=val_fraction/(1-train_fraction),
                           stratify=yy[hold],random_state=seed)
    return tuple(torch.tensor(x,dtype=torch.long) for x in (tr,va,te))

def load_node_npz(path,seed=0):
    """Required arrays: x [N,F], edge_index [2,M] or edges [M,2], y[N],
    sensitive[N] (optional; unknown=-1). Optional split arrays train_idx etc.
    Do not add labels/sensitive fields into x automatically. Caller audits leakage.
    """
    f=np.load(path,allow_pickle=False)
    for field in ('x','y'):
        if field not in f:raise ValueError(f'Missing {field} in {path}')
    x=torch.as_tensor(f['x'],dtype=torch.float32)
    y=torch.as_tensor(f['y'],dtype=torch.long).flatten()
    if not torch.isfinite(x).all():raise ValueError('Nonfinite node features; preprocess explicitly')
    if 'edges' in f: ee=f['edges']
    elif 'edge_index' in f: ee=f['edge_index'].T
    else:raise ValueError('Missing edges or edge_index')
    ee=np.asarray(ee,dtype=np.int64).reshape(-1,2)
    es=sorted({(min(int(u),int(v)),max(int(u),int(v))) for u,v in ee if u!=v})
    edges=torch.tensor(es,dtype=torch.long).reshape(-1,2)
    sensitive=torch.tensor(f['sensitive'],dtype=torch.long).flatten() if 'sensitive' in f else torch.full_like(y,-1)
    if all(k in f for k in ('train_idx','val_idx','test_idx')):
        ids=[torch.as_tensor(f[k],dtype=torch.long).flatten() for k in ('train_idx','val_idx','test_idx')]
    else:ids=node_splits(y.numpy(),seed=seed)
    return validate_node(NodeData(x,edges,y,sensitive,*ids))

class NodeGNN(nn.Module):
    """Reference dense GIN-like message passing with node logits, no label leakage.
    For scalability replace NxN adjacency with sparse scatter aggregation.
    """
    def __init__(self,input_dim,hidden=32,layers=2,kind='sage'):
        super().__init__()
        if kind not in ('sage','gcn','gin','mlp'):raise ValueError(kind)
        self.kind=kind
        self.proj=nn.Linear(input_dim,hidden)
        self.self_layers=nn.ModuleList(nn.Linear(hidden,hidden) for _ in range(layers))
        self.nbr_layers=nn.ModuleList(nn.Linear(hidden,hidden,bias=False) for _ in range(layers))
        self.out=nn.Linear(hidden,1)
    def forward(self,data):
        n=len(data.x);h=F.relu(self.proj(data.x));a=h.new_zeros((n,n))
        if self.kind!='mlp' and len(data.edges):
            u,v=data.edges.T;a[u,v]=1.;a[v,u]=1.
        for sl,nl in zip(self.self_layers,self.nbr_layers):
            if self.kind=='gin':h=F.relu(sl(h+a@h))
            elif self.kind=='gcn':
                at=a+torch.eye(n,device=h.device);d=at.sum(1).clamp(min=1).pow(-.5)
                h=F.relu(sl((d[:,None]*at*d[None,:])@h))
            elif self.kind=='sage':h=F.relu(sl(h)+nl((a@h)/a.sum(1,keepdim=True).clamp(min=1)))
            else:h=F.relu(sl(h))
        return self.out(h).flatten()

def ego_graph(data,target,hops=2):
    """Return computational k-hop subgraph with original->local node mapping.
    For full graph GNNs with >hops layers, set hops>=message-passing layers.
    """
    target=int(target)
    if target<0 or target>=len(data.x):raise ValueError('Invalid target')
    if hops<0:raise ValueError('hops must be nonnegative')
    seen={target};front={target}
    for _ in range(hops):
        nxt=set()
        for u,v in data.edges.tolist():
            if u in front:nxt.add(v)
            if v in front:nxt.add(u)
        front=nxt-seen;seen|=front
    nodes=sorted(seen);mp={v:i for i,v in enumerate(nodes)}
    edges=torch.tensor([(mp[u],mp[v]) for u,v in data.edges.tolist() if u in mp and v in mp],dtype=torch.long).reshape(-1,2)
    return Graph(data.x[nodes],edges,int(data.y[target]),int(data.sensitive[target])),mp[target],nodes

def synthetic_node_data(seed=0,n=90):
    """Smoke fixture only: protected field correlated with nuisance, labels from features.
    Not a realistic fairness ground-truth benchmark.
    """
    rng=np.random.default_rng(seed)
    x=rng.standard_normal((n,4)).astype('float32')
    s=(rng.random(n)>.5).astype('int64');x[:,3]+=.5*s
    y=(x[:,0]+.6*x[:,1]+.15*rng.standard_normal(n)>0).astype('int64')
    pairs=[(u,v) for u in range(n) for v in range(u+1,n) if rng.random()<.055]
    tr,va,te=node_splits(y,seed)
    return validate_node(NodeData(torch.tensor(x),torch.tensor(pairs,dtype=torch.long).reshape(-1,2),
                                   torch.tensor(y),torch.tensor(s),tr,va,te))
