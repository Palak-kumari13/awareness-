"""Held-out randomized motif benchmark and controlled selector training.

Purposefully isolates selection learning from fairness and robustness. Synthetic
motif labels describe a generator, not real-world causality.
"""
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from .core import Graph, GNN, validate
from .trainlib import seed_all


def randomized_motifs(n_graphs=180, seed=0, min_nodes=12, max_nodes=22,
                      background=(0.025,0.065), hard_negative=True):
    """Class 1 includes a planted triangle, class 0 an induced planted path.

    Motif locations and background edges are randomized and unrelated to label.
    Hard negatives may contain incidental background triangles; planted-edge
    recovery uses generator-provided masks, never labels/features as shortcuts.
    """
    rng=random.Random(seed)
    out=[]
    for index in range(n_graphs):
        y=index % 2
        n=rng.randint(min_nodes,max_nodes)
        nodes=rng.sample(range(n),3)
        a,b,c=nodes
        motif_set={tuple(sorted((a,b))),tuple(sorted((b,c)))}
        if y: motif_set.add(tuple(sorted((a,c))))
        edges=set(motif_set)
        p=rng.uniform(*background)
        for i in range(n):
            for j in range(i+1,n):
                # Induced construction: exclude motif endpoint closing edge in negatives.
                if (i,j)==tuple(sorted((a,c))) and not y: continue
                if (i,j) not in edges and rng.random()<p: edges.add((i,j))
        # Make hard negatives by planting another triangle in some label-0 graphs.
        # This intentionally makes the motif non-unique; no claims of perfect recovery.
        if hard_negative and y==0 and n>=8 and rng.random()<.35:
            others=[v for v in range(n) if v not in nodes]
            t=rng.sample(others,3)
            edges.update(tuple(sorted(pair)) for pair in ((t[0],t[1]),(t[1],t[2]),(t[0],t[2])))
        ordered=sorted(edges)
        motif=torch.tensor([e in motif_set for e in ordered],dtype=torch.bool)
        # No node positional identifiers: constant attributes across graphs.
        out.append(validate(Graph(torch.ones((n,1)),torch.tensor(ordered,dtype=torch.long).reshape(-1,2),y,None,motif)))
    rng.shuffle(out)
    return out


def frozen_split(seed=0,n_graphs=180):
    # Each split independently generated; training and evaluation generator streams differ.
    return (randomized_motifs(n_graphs,seed=seed*1000+101),
            randomized_motifs(max(40,n_graphs//4),seed=seed*1000+202),
            randomized_motifs(max(40,n_graphs//4),seed=seed*1000+303))


def soft_binary_kl(teacher_logits,student_logits):
    t=torch.sigmoid(teacher_logits.detach()).clamp(1e-5,1-1e-5)
    # BCE minus target entropy = KL(Bernoulli(t)||Bernoulli(s))
    return (F.binary_cross_entropy_with_logits(student_logits,t,reduction='none')
            +t*torch.log(t)+(1-t)*torch.log(1-t)).mean()


def train_selector_variant(model,trainset,valset,method='joint',reference=None,
                           epochs=60,lr=.001,seed=0,patience=15,distill_weight=.25):
    """Methods: joint, alternate (freeze scorer/predictor in alternating epochs), teacher.
    Teacher must be frozen and pre-trained on training subset only.
    No motif supervision or use of held-out test labels in fitting.
    """
    if method not in ('joint','alternate','teacher'):raise ValueError(method)
    if method=='teacher' and reference is None:raise ValueError('teacher model required')
    if not model.selector_on:raise ValueError('selector must be enabled')
    seed_all(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    best=None;best_score=float('inf');stale=0;history=[]
    for ep in range(epochs):
        model.train()
        if method=='alternate':
            # Even epochs update selector, odd epochs update predictor, then alternate.
            for param in model.parameters():param.requires_grad_(True)
            for param in (model.scores.parameters() if ep%2 else
                          list(model.project.parameters())+list(model.blocks.parameters())+list(model.out.parameters())):
                param.requires_grad_(False)
        else:
            for param in model.parameters():param.requires_grad_(True)
        idx=list(range(len(trainset)));random.Random(seed*100000+ep).shuffle(idx)
        total=0.
        for j in range(0,len(idx),16):
            chunk=[trainset[i] for i in idx[j:j+16]]
            logits=torch.stack([model(g) for g in chunk])
            ys=torch.tensor([g.y for g in chunk],dtype=logits.dtype)
            loss=F.binary_cross_entropy_with_logits(logits,ys)
            if method=='teacher':
                reference.eval()
                with torch.no_grad(): tlogits=torch.stack([reference(g) for g in chunk])
                loss=loss+distill_weight*soft_binary_kl(tlogits,logits)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            total+=float(loss.detach())*len(chunk)
        for param in model.parameters():param.requires_grad_(True)
        model.eval()
        with torch.no_grad():
            vl=torch.stack([model(g) for g in valset]);vy=torch.tensor([g.y for g in valset],dtype=vl.dtype)
            val_loss=float(F.binary_cross_entropy_with_logits(vl,vy))
        history.append({'epoch':ep,'train_loss':total/len(trainset),'val_bce':val_loss})
        if val_loss<best_score-1e-6:
            best_score=val_loss;best=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(best)
    return model,history


def explain_precision(model,graphs,random_repeats=10,seed=0):
    model.eval();rng=random.Random(seed); learned=[]; randoms=[]; score_spreads=[]
    with torch.no_grad():
        for g in graphs:
            d=model(g,return_details=True)
            chosen=d['mask'].bool(); k=int(chosen.sum());m=len(g.edges)
            if not k or g.motif is None:continue
            learned.append(float(g.motif[chosen].float().mean()))
            randoms.append(np.mean([float(g.motif[torch.tensor(rng.sample(range(m),k))].float().mean()) for _ in range(random_repeats)]))
            score_spreads.append(float(d['scores'].std(unbiased=False)))
    return {'motif_precision':float(np.mean(learned)) if learned else None,
            'random_precision':float(np.mean(randoms)) if randoms else None,
            'score_std':float(np.mean(score_spreads)) if score_spreads else None,
            'n_graphs':len(learned)}
