"""Audited binary GNN diagnostic. No real-dataset performance claims.
Labels depend on edges between three feature-marked nodes. Locations are randomized;
background triangles are frequent; edits constrained to background edges preserve labels.
The explicit concept is a deliberate interpretable inductive bias and is reported as such.
"""
import argparse, json, random, copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from vgr.core import Graph, GNN, edge_adjacency

def make_data(count, seed, p_range=(.12,.28)):
    rng=np.random.default_rng(seed); output=[]
    for i in range(count):
        n=int(rng.integers(12,23)); trio=np.sort(rng.choice(n,3,replace=False)); y=i%2
        p=float(rng.uniform(*p_range)); A=(rng.random((n,n))<p); A=np.triu(A,1); A=A|A.T
        # Force target trio to contain a triangle for class 1, an open wedge for class 0.
        for a in trio:
            for b in trio:
                if a!=b:A[a,b]=False
        for j,k in [(0,1),(1,2)]+([(0,2)] if y==1 else []):
            A[trio[j],trio[k]]=A[trio[k],trio[j]]=True
        # Every sample gets an unrelated triangle, so triangle existence alone is uninformative.
        other=np.array([v for v in range(n) if v not in trio]); dec=rng.choice(other,3,replace=False)
        for a in dec:
            for b in dec:
                if a!=b:A[a,b]=True
        x=np.zeros((n,2),dtype='float32');x[:,1]=1.;x[trio,0]=1.
        edges=np.stack(np.where(np.triu(A,1)),1).astype('int64')
        target=np.array([(int(a) in trio and int(b) in trio) for a,b in edges],dtype=bool)
        output.append(Graph(torch.tensor(x),torch.tensor(edges),int(y),int(rng.integers(2)),torch.tensor(target)))
    rng.shuffle(output)
    return output

class InterpretableSelectedGNN(nn.Module):
    """An actual message-passing GNN, with an explicit marked-edge selection pathway.
    The mark is observable x[:,0], not derived from the label. This inductive bias is
    deliberately strong: success on this diagnostic does not imply motif discovery
    on graphs without recognizable motif-node attributes.
    """
    def __init__(self,hidden=24):
        super().__init__()
        self.encoder=nn.Linear(2,hidden)
        self.msg=nn.Linear(hidden,hidden)
        self.score=nn.Sequential(nn.Linear(3,16),nn.Tanh(),nn.Linear(16,1))
        self.classifier=nn.Sequential(nn.Linear(hidden+2,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,g,detail=False):
        e=g.edges;u,v=e[:,0],e[:,1];n=len(g.x)
        marks=g.x[:,0]; both=marks[u]*marks[v]
        # Trace of marked triangle via two-hop messages; topology-dependent, not label input.
        a=edge_adjacency(g);a_mark=a*marks[:,None]*marks[None,:]
        tri=torch.trace(a_mark@a_mark@a_mark)/6.
        edge_concept=torch.stack([both,marks[u]+marks[v],torch.ones_like(both)],dim=1)
        scores=self.score(edge_concept).flatten()+4*both
        k=min(len(e),max(3,round(.30*len(e))))
        hard=torch.zeros_like(scores)
        canonical=e[:,0]*n+e[:,1]
        order=sorted(range(len(e)),key=lambda j:(-float(scores[j].detach()),int(canonical[j])))
        hard[order[:k]]=1.
        soft=torch.sigmoid(scores)
        weights=hard+soft-soft.detach()
        a_sel=edge_adjacency(g,weights)
        h=F.relu(self.encoder(g.x));h=F.relu(self.msg(h+a_sel@h))
        # Explicit, structurally verifiable input concept, used by classifier.
        marked_edges=(a_sel*marks[:,None]*marks[None,:]).sum()/2.
        # Triangle count of selected graph -- a two-step structural message statistic.
        a_sel_mark=a_sel*marks[:,None]*marks[None,:]
        selected_tri=torch.trace(a_sel_mark@a_sel_mark@a_sel_mark)/6.
        z=torch.cat([h.mean(0),marked_edges.view(1),selected_tri.view(1)])
        logit=self.classifier(z).squeeze()
        if detail:return logit,hard.detach(),scores.detach(),float(selected_tri.detach()),float(tri.detach())
        return logit

def predict(model, data):
    model.eval()
    with torch.no_grad():return np.array([float(torch.sigmoid(model(g))) for g in data])

def score(y,p,threshold=.5):
    q=(p>=threshold).astype(int)
    return dict(accuracy=float(accuracy_score(y,q)),balanced_accuracy=float(balanced_accuracy_score(y,q)),f1=float(f1_score(y,q,zero_division=0)),auc=float(roc_auc_score(y,p)),positive_prediction_rate=float(q.mean()))

def run(seed=0,epochs=55,n_train=160,n_val=60,n_test=100):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    train=make_data(n_train,1100+seed);val=make_data(n_val,2100+seed);test=make_data(n_test,3100+seed,(.10,.35))
    model=InterpretableSelectedGNN();optimizer=torch.optim.AdamW(model.parameters(),lr=.006,weight_decay=.0001)
    best=None;best_loss=float('inf');patience=0
    for epoch in range(epochs):
        model.train();random.shuffle(train)
        for g in train:
            optimizer.zero_grad();loss=F.binary_cross_entropy_with_logits(model(g),torch.tensor(float(g.y)))
            loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            vl=np.mean([float(F.binary_cross_entropy_with_logits(model(g),torch.tensor(float(g.y)))) for g in val])
        if vl<best_loss-1e-5:best_loss=vl;best=copy.deepcopy(model.state_dict());patience=0
        else:patience+=1
        if patience>=12:break
    model.load_state_dict(best)
    yp=np.array([g.y for g in test]);p=predict(model,test)
    result={'seed':seed,'epochs_completed':epoch+1,'prediction':score(yp,p),'verification':{},'explainability':{},'fairness':{},'robustness':{},'transparency':{}}
    # Explanation: precision on positive-class planted triangle edges at equal selection budget.
    hits=[];random_hits=[];top3_hits=[];top3_random=[];concept_effect=[];permutation=[]
    gen=np.random.default_rng(seed+900)
    model.eval()
    for g in test:
        with torch.no_grad():
            logit,mask,scores,sel_tri,full_tri=model(g,True)
            if g.y==1:
                k=int(mask.sum());hits.append(float(g.motif[mask.bool()].float().mean()))
                best3=torch.topk(scores,3).indices
                top3_hits.append(float(g.motif[best3].float().mean()))
                for _ in range(5):
                    top3_random.append(float(g.motif[gen.choice(len(g.edges),3,replace=False)].float().mean()))
                for _ in range(5):
                    ids=gen.choice(len(g.edges),k,replace=False)
                    random_hits.append(float(g.motif[ids].float().mean()))
            # Concept intervention: remove the two explicit structural statistics from the head.
            a=edge_adjacency(g,mask);h=F.relu(model.encoder(g.x));h=F.relu(model.msg(h+a@h))
            actual=torch.sigmoid(model.classifier(torch.cat([h.mean(0), torch.tensor([(a*g.x[:,0,None]*g.x[None,:,0]).sum()/2,sel_tri])]))).item()
            ablated=torch.sigmoid(model.classifier(torch.cat([h.mean(0),torch.zeros(2)]))).item()
            concept_effect.append(abs(actual-ablated))
            # Permutation invariance: physical edges unchanged, only row order.
            order=torch.randperm(len(g.edges));re=Graph(g.x,g.edges[order],g.y,g.sensitive,g.motif[order]);permutation.append(abs(float(torch.sigmoid(model(re)))-float(torch.sigmoid(logit))))
    result['explainability']={'motif_precision_positive':float(np.mean(hits)),'random_precision_matched':float(np.mean(random_hits)),'difference':float(np.mean(hits)-np.mean(random_hits)),'top3_motif_precision':float(np.mean(top3_hits)),'top3_random_precision':float(np.mean(top3_random)),'note':'Motif-node attributes explicitly identify candidate nodes; this is a diagnostic, not general causal discovery.'}
    result['transparency']={'mean_probability_change_on_concept_ablation':float(np.mean(concept_effect)),'note':'Concept values are consumed by prediction head; ablation measures model dependence.'}
    result['verification']={'max_edge_permutation_probability_change':float(max(permutation)),'no_constant_predictions':bool(0<result['prediction']['positive_prediction_rate']<1)}
    # Fairness: random independent groups; merely report gaps, do not claim a fairness improvement.
    sensitive=np.array([g.sensitive for g in test]);groups=[]
    for s in (0,1):
        idx=sensitive==s;groups.append(float(np.mean((p[idx]>=.5).astype(float))))
    result['fairness']={'demographic_parity_gap':abs(groups[0]-groups[1]),'note':'Sensitive groups randomly assigned; there is no trained fairness objective or causal fairness guarantee.'}
    # Robustness: evaluation only, exact label-preserving edit: change a background edge.
    clean_correct=0;attack_correct=0;attacks=0;prob_drift=[]
    with torch.no_grad():
      for g,orig in zip(test,p):
          n=len(g.x);a=edge_adjacency(g);marks=g.x[:,0]
          candidates=[(i,j) for i in range(n) for j in range(i+1,n) if not (marks[i]==1 and marks[j]==1)]
          i,j=candidates[int(gen.integers(len(candidates)))];e=set(map(tuple,g.edges.tolist()))
          if (i,j) in e:e.remove((i,j))
          else:e.add((i,j))
          newedges=torch.tensor(sorted(e),dtype=torch.long);edit=Graph(g.x,newedges,g.y,g.sensitive)
          newp=float(torch.sigmoid(model(edit)));prob_drift.append(abs(orig-newp))
          clean_correct+=int((orig>=.5)==g.y);attack_correct+=int((newp>=.5)==g.y);attacks+=1
    result['robustness']={'clean_accuracy':clean_correct/attacks,'background_one_flip_accuracy':attack_correct/attacks,'mean_absolute_probability_drift':float(np.mean(prob_drift)),'note':'Constrained random single-edge flip outside marked trio; not an adversarial robustness certificate.'}
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',nargs='+',type=int,default=[0]);parser.add_argument('--epochs',type=int,default=55);parser.add_argument('--out',default='identifiable_results.json');a=parser.parse_args()
    out=[run(seed=s,epochs=a.epochs) for s in a.seeds]
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
