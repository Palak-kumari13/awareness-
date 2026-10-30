"""Independent awareness models and reproducible binary social graph comparison.
Not a fairness benchmark. The synthetic task is a controlled structural experiment.
"""
import argparse,copy,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score
from vgr.awareness import structural_features
from vgr.core import Graph

class AwareSelectedGNN(nn.Module):
    def __init__(self,mode='topology',hidden=32,retention=.6,select=True):
        super().__init__()
        self.mode=mode;self.retention=retention;self.select=select
        dim={'topology':6,'motif':7,'spectral':7,'hybrid':11}[mode]
        self.node=nn.Sequential(nn.Linear(4,hidden),nn.ReLU())
        self.edge_score=nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.message=nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(2)])
        self.readout=nn.Sequential(nn.Linear(hidden*2+3,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,g,detail=False):
        n=g.x.shape[0];e=g.edges;local,ef,a=structural_features(e,n,self.mode)
        if local.shape[1]>3:local=local[:,:3]
        h=self.node(torch.cat([g.x[:,:1],local],-1))
        scores=self.edge_score(ef).flatten()
        hard=torch.ones_like(scores)
        if self.select and len(scores):
            k=max(1,min(len(scores),round(self.retention*len(scores))))
            # canonical edge tie-break; no dependence on list ordering
            keys=e[:,0]*n+e[:,1]
            ids=sorted(range(len(scores)),key=lambda i:(-float(scores[i].detach()),int(keys[i])))[:k]
            hard=torch.zeros_like(scores);hard[ids]=1
        soft=torch.sigmoid(scores)
        w=hard+soft-soft.detach() if self.select else hard
        selected=torch.zeros((n,n),device=a.device)
        if len(e):
            selected[e[:,0],e[:,1]]=w;selected[e[:,1],e[:,0]]=w
        d=selected.sum(-1).clamp_min(1)
        for layer in self.message:
            h=F.relu(layer(h+selected@h/d[:,None]))
        # all graph-level concept inputs depend exclusively on selected adjacency
        sdeg=selected.sum(-1);s2=selected@selected
        tri=torch.trace(s2@selected)/6
        concepts=torch.stack([sdeg.mean()/max(n-1,1),tri/max(n*n,1), ((sdeg*(sdeg-1)/2).clamp_min(0)).mean()/max(n*n,1)])
        z=torch.cat([h.mean(0),h.max(0).values,concepts])
        logit=self.readout(z).squeeze()
        if detail:return logit,hard.detach(),scores.detach(),concepts.detach()
        return logit

def social_data(count,seed):
    rng=np.random.default_rng(seed);out=[]
    for i in range(count):
        y=i%2;n=int(rng.integers(18,31));groups=rng.permutation(np.r_[np.zeros(n//2,dtype=int),np.ones(n-n//2,dtype=int)])
        # SBM with different inter/intra affinity; no group or class markers in features.
        p_in,p_out=((.42,.07) if y==1 else (.08,.41))
        a=np.zeros((n,n),dtype=bool)
        for u in range(n):
            for v in range(u+1,n):
                p=(p_in if groups[u]==groups[v] else p_out)
                if rng.random()<p:a[u,v]=a[v,u]=True
        edges=np.stack(np.where(np.triu(a,1)),axis=1).astype(np.int64)
        # No group/label leakage. One constant node feature.
        out.append(Graph(torch.ones((n,1)),torch.as_tensor(edges),int(y),0))
    rng.shuffle(out);return out

def evaluate(model,data):
    model.eval()
    with torch.no_grad():p=np.array([torch.sigmoid(model(g)).item() for g in data])
    y=np.array([g.y for g in data]);z=(p>=.5).astype(int)
    return {'accuracy':float(accuracy_score(y,z)),'balanced_accuracy':float(balanced_accuracy_score(y,z)),'auroc':float(roc_auc_score(y,p)),'f1':float(f1_score(y,z,zero_division=0)),'positive_rate':float(z.mean()),'prob_std':float(p.std())}

def run(mode,seed,epochs=20,train_size=100,val_size=40,test_size=60,select=True):
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    tr=social_data(train_size,1000+seed);va=social_data(val_size,2000+seed);te=social_data(test_size,3000+seed)
    model=AwareSelectedGNN(mode,select=select);optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    best=float('inf');best_state=None;patience=0
    for epoch in range(epochs):
        random.shuffle(tr);model.train()
        for g in tr:
            optimizer.zero_grad();loss=F.binary_cross_entropy_with_logits(model(g),torch.tensor(float(g.y)))
            loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():v=float(np.mean([F.binary_cross_entropy_with_logits(model(g),torch.tensor(float(g.y))).item() for g in va]))
        if v<best-1e-4:best=v;best_state=copy.deepcopy(model.state_dict());patience=0
        else:patience+=1
        if patience>=7:break
    model.load_state_dict(best_state)
    results=evaluate(model,te)
    spreads=[];ranked=[];randoms=[];generator=np.random.default_rng(9000+seed)
    with torch.no_grad():
        for g in te[:min(20,len(te))]:
            logit,mask,scores,concepts=model(g,True);spreads.append(float(scores.std(unbiased=False)) if len(scores)>1 else 0.)
            if len(g.edges)<3:continue
            k=min(3,len(g.edges));top=torch.argsort(scores,descending=True)[:k]
            def remove(ids):
                keep=torch.ones(len(g.edges),dtype=torch.bool);keep[ids]=False
                ge=Graph(g.x,g.edges[keep],g.y,g.sensitive)
                return float(torch.sigmoid(model(ge)))
            base=float(torch.sigmoid(logit));ranked.append(abs(base-remove(top)))
            for _ in range(3):
                ids=torch.as_tensor(generator.choice(len(g.edges),k,replace=False));randoms.append(abs(base-remove(ids)))
    results.update({'score_spread_mean':float(np.mean(spreads)),'ranked_edge_removal_drift':float(np.mean(ranked)),'random_edge_removal_drift':float(np.mean(randoms)),'epochs':epoch+1})
    return results

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);pa.add_argument('--epochs',type=int,default=20);pa.add_argument('--out',default='outputs/awareness_comparison.json');pa.add_argument('--train-size',type=int,default=100);pa.add_argument('--val-size',type=int,default=40);pa.add_argument('--test-size',type=int,default=60);a=pa.parse_args()
    rows=[]
    for mode in ('topology','motif','spectral','hybrid'):
        for seed in a.seeds:
            r=run(mode,seed,a.epochs,a.train_size,a.val_size,a.test_size)
            rows.append(dict(variant=mode,seed=seed,**r));print(mode,seed,r,flush=True)
    Path(a.out).parent.mkdir(exist_ok=True,parents=True);Path(a.out).write_text(json.dumps(rows,indent=2))
