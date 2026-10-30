"""Small-graph research reference implementation. One graph per forward call.
Undirected simple graphs: one unique edge per unordered node pair.
Hard top-k selection is exact in the forward pass and uses a biased ST gradient.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class Graph:
    x: torch.Tensor
    edges: torch.Tensor  # shape [M,2], unique unordered u<v
    y: int
    sensitive: int | None = None
    motif: torch.Tensor | None = None

    def to(self, device):
        return Graph(self.x.to(device), self.edges.to(device), self.y, self.sensitive,
                     None if self.motif is None else self.motif.to(device))


def validate(g):
    assert g.x.ndim == 2 and g.x.size(0) > 0
    assert g.edges.ndim == 2 and g.edges.size(1) == 2
    assert g.edges.dtype == torch.long
    if g.edges.numel():
        assert bool(((g.edges >= 0) & (g.edges < len(g.x))).all())
        assert bool((g.edges[:,0] < g.edges[:,1]).all())
        assert len(torch.unique(g.edges,dim=0)) == len(g.edges)
    assert g.y in (0,1)
    return g


def edge_adjacency(g, weights=None):
    n = len(g.x)
    a = g.x.new_zeros((n,n))
    if len(g.edges):
        w = g.x.new_ones(len(g.edges)) if weights is None else weights
        # index_put with accumulate supports gradients through edge weights
        a = a.index_put((g.edges[:,0],g.edges[:,1]),w,accumulate=True)
        a = a + a.T
    return a


def concepts(g, adj):
    """Transparent structural channels actually consumed by classifier.
    Differentiable mean degree, wedge density, and triangle density, with
    fixed scaling based on number of nodes. Not causal concepts.
    """
    n = len(g.x)
    degree = adj.sum(-1)
    denom = max(n-1,1)
    return torch.stack((degree.mean()/denom,
        ((degree*(degree-1)).clamp(min=0)).mean()/max(denom*(denom-1),1),
        torch.trace(adj @ adj @ adj)/max(n*(n-1)*(n-2),1)))


class GNN(nn.Module):
    def __init__(self, input_dim, hidden=32, layers=2, selector=True, budget=.5, concepts_on=True):
        super().__init__()
        self.selector_on=selector
        self.budget=budget
        self.concepts_on=concepts_on
        self.project=nn.Linear(input_dim,hidden)
        self.scores=nn.Sequential(nn.Linear(hidden*2+2,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden)) for _ in range(layers)])
        self.out=nn.Sequential(nn.Linear(hidden+(3 if concepts_on else 0),hidden),nn.ReLU(),nn.Linear(hidden,1))

    def edge_scores(self,g):
        h=self.project(g.x)
        u,v=g.edges[:,0],g.edges[:,1]
        a=edge_adjacency(g)
        d=a.sum(1)/max(len(g.x)-1,1)
        inp=torch.cat([h[u]+h[v],torch.abs(h[u]-h[v]),(d[u]+d[v]).unsqueeze(1),(d[u]-d[v]).abs().unsqueeze(1)],1)
        return self.scores(inp).flatten()

    def select(self,g):
        m=len(g.edges)
        if not self.selector_on or m==0:
            w=g.x.new_ones(m)
            return w,w.detach(),g.x.new_zeros(m)
        scores=self.edge_scores(g)
        k=min(m,max(1,round(self.budget*m)))
        # Tie-breaking depends only on canonical edge IDs, not original input order.
        canonical=g.edges[:,0]*len(g.x)+g.edges[:,1]
        order=sorted(range(m),key=lambda j:(-float(scores[j].detach()),int(canonical[j])))
        hard=torch.zeros_like(scores)
        hard[order[:k]]=1.
        soft=torch.sigmoid(scores)
        # exact hard edges in forward, nonzero surrogate derivative backward
        weight=hard+(soft-soft.detach())
        return weight,hard,scores

    def forward(self,g, edge_weights=None, return_details=False):
        validate(g)
        if edge_weights is None:
            w,hard,score=self.select(g)
        else:
            w=edge_weights
            hard=(w.detach()>0.5).float()
            score=self.edge_scores(g) if self.selector_on and len(g.edges) else g.x.new_zeros(len(g.edges))
        adj=edge_adjacency(g,w)
        h=F.relu(self.project(g.x))
        for block in self.blocks:
            # GIN-like propagation; no normalized-adjacency artifacts from zero-weight edges.
            h=F.relu(block(h+adj @ h))
        pooled=h.mean(0)
        c=concepts(g,adj)
        rep=torch.cat((pooled,c)) if self.concepts_on else pooled
        logit=self.out(rep).squeeze()
        if return_details:
            return {'logit':logit,'prob':torch.sigmoid(logit),'mask':hard,'scores':score,
                    'concepts':c,'representation':rep}
        return logit


def selected_graph(g, mask):
    return Graph(g.x,g.edges[mask.bool()],g.y,g.sensitive,
                 None if g.motif is None else g.motif[mask.bool()])


def intervention(g, drop_indices):
    keep=torch.ones(len(g.edges),dtype=torch.bool,device=g.edges.device)
    if len(drop_indices):
        keep[torch.as_tensor(drop_indices,dtype=torch.long,device=keep.device)]=False
    return selected_graph(g,keep)
