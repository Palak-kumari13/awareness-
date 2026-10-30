"""Experimental unified selector / predictor branches. Small graphs only.
Correct hard forward edge gating with a biased straight-through gradient.
These implementations do not claim certified fairness, robustness or causality.
"""
import copy, random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from .core import Graph, GNN, edge_adjacency
from .node import NodeGNN, NodeData
from .fairness import dp_surrogate, eo_surrogate


def planted_multilabel(n=180,seed=0):
    """Two independent planted features: triangle on nodes 0,1,2; 4-cycle on 3,4,5,6.
    No label-derived node features. Base edges avoid intra-motif accidental patterns.
    Ground truth indicates generator-inserted edges, not real-world causality.
    """
    rng=random.Random(seed); out=[]
    for i in range(n):
        a=i%2;b=(i//2)%2
        size=rng.randrange(13,19);es=set();ground=set()
        for u in range(7,size):
            for v in range(u+1,size):
                if rng.random()<.11:es.add((u,v))
        for e in [(2,7),(6,8)]:es.add(e)
        if a:
            for e in [(0,1),(1,2),(0,2)]:es.add(e);ground.add(e)
        else:
            es.update([(0,1),(1,2)])
        if b:
            for e in [(3,4),(4,5),(5,6),(3,6)]:es.add(e);ground.add(e)
        else:es.update([(3,4),(4,5),(5,6)])
        ed=sorted(es); motif=torch.tensor([e in ground for e in ed],dtype=torch.bool)
        # separate graph-level multilabel target, Graph.y is a legacy scalar placeholder
        out.append((Graph(torch.ones(size,1),torch.tensor(ed,dtype=torch.long),int(a),motif=motif),
                    torch.tensor([a,b],dtype=torch.float32)))
    rng.shuffle(out)
    return out

class MultiLabelGNN(GNN):
    def __init__(self,input_dim,labels=2,**kw):
        super().__init__(input_dim,**kw)
        h=self.out[0].in_features
        hidden=self.out[0].out_features
        self.out=nn.Sequential(nn.Linear(h,hidden),nn.ReLU(),nn.Linear(hidden,labels))


def train_graph_multilabel(samples,seed=0,epochs=60,lr=.003,hidden=32,budget=.65,
                           consistency=.1,selector=True):
    """Train-only random edge deletion. Stratified split by label-combination is caller's duty.
    Selection retains hard top-k edges in forward pass. Independent BCE per label.
    """
    torch.manual_seed(seed);rng=random.Random(seed)
    model=MultiLabelGNN(samples[0][0].x.shape[1],labels=len(samples[0][1]),hidden=hidden,
                        selector=selector,budget=budget)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    history=[]
    for ep in range(epochs):
        model.train();order=list(range(len(samples)));rng.shuffle(order);running=0
        for j in order:
            g,y=samples[j];opt.zero_grad()
            z=model(g).reshape(-1)
            loss=F.binary_cross_entropy_with_logits(z,y)
            if consistency and len(g.edges)>4:
                # independent benign edit training is an assumption, not label-preserving ground truth
                idx=[k for k in range(len(g.edges)) if rng.random()>.06]
                if not idx:idx=[0]
                sub=Graph(g.x,g.edges[idx],g.y,motif=None)
                z2=model(sub).reshape(-1)
                loss=loss+consistency*F.mse_loss(z.sigmoid(),z2.sigmoid())
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step();running+=float(loss.detach())
        history.append(running/len(samples))
    return model,history

@torch.no_grad()
def evaluate_multilabel(model,samples,thresholds=None):
    model.eval();p=np.array([model(g).sigmoid().cpu().numpy() for g,y in samples]);
    y=np.array([yy.numpy() for g,yy in samples]).astype(int)
    if thresholds is None:thresholds=np.full(y.shape[1],.5)
    preds=(p>=np.asarray(thresholds)).astype(int)
    per=[]
    for j in range(y.shape[1]):
        per.append({'auroc':float(roc_auc_score(y[:,j],p[:,j])) if len(set(y[:,j]))==2 else None,
                    'f1':float(f1_score(y[:,j],preds[:,j],zero_division=0)),
                    'positive_rate':float(preds[:,j].mean()),
                    'balanced_accuracy':float(balanced_accuracy_score(y[:,j],preds[:,j])) if len(set(y[:,j]))==2 else None})
    return {'per_label':per,'macro_f1':float(f1_score(y,preds,average='macro',zero_division=0)),
            'micro_f1':float(f1_score(y,preds,average='micro',zero_division=0)),
            'subset_accuracy':float(np.all(y==preds,axis=1).mean()), 'n':len(y)}

class SelectedNodeGNN(nn.Module):
    """Hard selected subgraph directly used for node prediction.
    Score depends on input features and degree; predicts ALL nodes simultaneously.
    Dense O(N^2), intended for small benchmarks, not Pokec.
    """
    def __init__(self,input_dim,hidden=32,budget=.6,layers=2):
        super().__init__();self.budget=budget
        self.project=nn.Linear(input_dim,hidden)
        self.scorer=nn.Sequential(nn.Linear(hidden*2+2,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.self_layers=nn.ModuleList(nn.Linear(hidden,hidden) for _ in range(layers))
        self.nbr_layers=nn.ModuleList(nn.Linear(hidden,hidden,bias=False) for _ in range(layers))
        self.out=nn.Linear(hidden,1)
    def select(self,d):
        e=d.edges;m=len(e)
        if m==0:return d.x.new_zeros(0),d.x.new_zeros(0)
        n=len(d.x);h=F.relu(self.project(d.x));deg=torch.bincount(e.flatten(),minlength=n).float()/max(n-1,1)
        u,v=e.T
        features=torch.cat((h[u]+h[v],torch.abs(h[u]-h[v]),(deg[u]+deg[v])[:,None],(deg[u]-deg[v]).abs()[:,None]),1)
        score=self.scorer(features).flatten();k=max(1,min(m,round(self.budget*m)))
        ids=(u*n+v).detach().cpu().tolist()
        order=sorted(range(m),key=lambda j:(-float(score[j].detach()),int(ids[j])))
        hard=torch.zeros_like(score);hard[order[:k]]=1
        w=hard+(torch.sigmoid(score)-torch.sigmoid(score).detach())
        return w,score
    def forward(self,d,return_details=False,return_embeddings=False):
        w,score=self.select(d);n=len(d.x);a=d.x.new_zeros((n,n))
        if len(d.edges):
            u,v=d.edges.T;a=a.index_put((u,v),w,accumulate=True);a=a+a.T
        h=F.relu(self.project(d.x))
        for sl,nl in zip(self.self_layers,self.nbr_layers):
            h=F.relu(sl(h)+nl((a@h)/a.sum(1,keepdim=True).clamp(min=1)))
        logits=self.out(h).flatten()
        if return_embeddings: return logits,h
        return (logits,{'edge_scores':score,'hard_selected':(w.detach()>.5).float()}) if return_details else logits


def train_selected_node(data,seed=0,epochs=120,lr=.005,budget=.6,fair_weight=0.,
                        fair_kind='dp',robust_weight=0.,hidden=32,angle_weight=0.):
    """Fairness computed on TRAIN nodes only, no test leakage.
    Robustness trained on random edge deletion, independent of greedy evaluation attacks.
    """
    torch.manual_seed(seed);rng=random.Random(seed)
    d=data;model=SelectedNodeGNN(d.x.size(1),hidden=hidden,budget=budget)
    opt=torch.optim.Adam(model.parameters(),lr=lr);best=None;best_val=float('inf');hist=[]
    if fair_weight and (d.sensitive[d.train_idx]<0).any():raise ValueError('Missing training sensitive attributes')
    for ep in range(epochs):
        model.train();opt.zero_grad();logits=model(d);idx=d.train_idx
        label=d.y[idx].float();n0=int((label==0).sum());n1=int((label==1).sum())
        pw=torch.tensor(max(1,n0)/max(1,n1),dtype=torch.float32)
        loss=F.binary_cross_entropy_with_logits(logits[idx],label,pos_weight=pw)
        if fair_weight:
            p=logits[idx].sigmoid();s=d.sensitive[idx];y=d.y[idx]
            loss=loss+fair_weight*(dp_surrogate(p,s) if fair_kind=='dp' else eo_surrogate(p,y,s))
        if robust_weight and len(d.edges)>3:
            keep=torch.tensor([rng.random()>.05 for _ in range(len(d.edges))],dtype=torch.bool)
            d2=NodeData(d.x,d.edges[keep],d.y,d.sensitive,d.train_idx,d.val_idx,d.test_idx)
            z2=model(d2)
            loss=loss+robust_weight*F.mse_loss(logits[idx].sigmoid(),z2[idx].sigmoid())
        if angle_weight:
            from .anglefair import anglefair_training_step
            loss,_=anglefair_training_step(model,d,weight=angle_weight,base_loss=loss)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        model.eval()
        with torch.no_grad():
            val=F.binary_cross_entropy_with_logits(model(d)[d.val_idx],d.y[d.val_idx].float()).item()
        if val<best_val:best_val=val;best=copy.deepcopy(model.state_dict())
        hist.append({'epoch':ep,'train_loss':float(loss.detach()),'val_bce':val})
    model.load_state_dict(best)
    return model,hist


def report_node(model,d,threshold=.5):
    from .fairness import node_metrics
    return {part:node_metrics(model,d,part,threshold=threshold) for part in ['train','val','test']}


def selector_gradient_norm(model,g):
    model.zero_grad();model(g).backward();return float(sum(p.grad.abs().sum().item() for p in model.scores.parameters() if p.grad is not None))
