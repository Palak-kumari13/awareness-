"""Phase 6: honest, finite-budget, undirected graph edge-flip evasion attacks.

Frozen trained model; no retraining or poisoning. Graph features/labels do not
change. No claim of semantic label preservation on real data. Model-aware greedy
attack uses the true label as its evaluation objective, not as model input.
"""
from dataclasses import dataclass
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from .core import Graph, validate
from .node import NodeData, validate_node


def canonical_pairs(edges):
    return {tuple(int(z) for z in e) for e in edges.detach().cpu().tolist()}


def candidate_pairs(n, allowed_nodes=None):
    nodes = sorted(set(range(n)) if allowed_nodes is None else set(int(x) for x in allowed_nodes))
    return [(u,v) for i,u in enumerate(nodes) for v in nodes[i+1:]]


def edit_edges(edges, pair, n, device):
    """Flip one unordered pair and return sorted unique u<v edges."""
    u,v=map(int,pair)
    if not 0<=u<v<n:raise ValueError('Invalid undirected edge pair')
    es=canonical_pairs(edges)
    if (u,v) in es: es.remove((u,v))
    else:es.add((u,v))
    return torch.tensor(sorted(es),dtype=torch.long,device=device).reshape(-1,2)


def with_graph_edges(g,edges):
    # motif flags are invalid after topology edits: never pass through stale flags.
    z=Graph(g.x,edges,g.y,g.sensitive,None)
    return validate(z)


def with_node_edges(d,edges):
    z=NodeData(d.x,edges,d.y,d.sensitive,d.train_idx,d.val_idx,d.test_idx)
    return validate_node(z)


def attack_budget(num_edges, rate=None, flips=None):
    if (rate is None)==(flips is None):raise ValueError('Specify exactly one of rate or flips')
    if flips is not None:
        if int(flips)!=flips or flips<0:raise ValueError('flips must be a nonnegative integer')
        return int(flips)
    if not math.isfinite(float(rate)) or not 0<=rate<=1:raise ValueError('rate must be in [0,1]')
    return int(round(rate*num_edges))


def _objective(logit,label):
    return float(F.binary_cross_entropy_with_logits(logit.reshape(()),torch.tensor(float(label),device=logit.device)).item())


def graph_attack(model,g,method='greedy',flips=1,seed=0,max_candidates=5000):
    """Return attacked graph, audit. Greedy maximizes supervised BCE sequentially.
    Evaluates each candidate by running the frozen model. Hard selection is
    recomputed for each candidate. Prevents toggling a pair more than once.
    'random' is an unweighted same-budget control.
    """
    validate(g)
    if method not in ('greedy','random'):raise ValueError(method)
    if flips<0 or int(flips)!=flips:raise ValueError('Invalid flip budget')
    pool=candidate_pairs(len(g.x)); budget=int(flips)
    if budget>len(pool):raise ValueError('Budget exceeds eligible pairs')
    if method=='greedy' and len(pool)>max_candidates:raise ValueError('Candidate space too large; set explicit max_candidates or use sampled evaluation')
    previous=model.training;model.eval(); current=g; edits=[];rng=random.Random(seed)
    try:
        with torch.no_grad():
            before=float(torch.sigmoid(model(current)).item())
            for _ in range(budget):
                if method=='random':pair=rng.choice(pool)
                else:
                    highest=-math.inf;pair=None
                    for p in pool:
                        cand=with_graph_edges(current,edit_edges(current.edges,p,len(current.x),current.edges.device))
                        score=_objective(model(cand),g.y)
                        if score>highest+1e-12:highest=score;pair=p
                current=with_graph_edges(current,edit_edges(current.edges,pair,len(current.x),current.edges.device))
                pool.remove(pair);edits.append(pair)
            after=float(torch.sigmoid(model(current)).item())
    finally:model.train(previous)
    # exactly the selected pair set changed, and all node features remain identical
    assert len(canonical_pairs(g.edges)^canonical_pairs(current.edges))==budget
    return current,{'method':method,'budget':budget,'edits':edits,'p_before':before,'p_after':after,'label':int(g.y),
        'pred_before':int(before>=.5),'pred_after':int(after>=.5),'delta_abs':abs(after-before),
        'attack_success':bool((after>=.5)!=(g.y==1)),'initial_correct':bool((before>=.5)==(g.y==1)),
        'label_preservation':'not verified; true label carried unchanged for evaluation only'}


def node_attack(model,data,target,method='greedy',flips=1,seed=0,hops=2,max_candidates=5000):
    """Evasion attack on pairs inside target's fixed ORIGINAL ego node set.
    Uses whole-graph node predictor so changes are evaluated in full context.
    For deeper GNNs set hops accordingly. Other node features and split IDs fixed.
    """
    validate_node(data)
    if method not in ('greedy','random'):raise ValueError(method)
    target=int(target)
    if target<0 or target>=len(data.x):raise ValueError('Invalid target')
    if flips<0 or int(flips)!=flips:raise ValueError('Invalid flip budget')
    # find original k-hop nodes, without relabeling original data
    seen={target};front={target}; edge_list=data.edges.detach().cpu().tolist()
    for _ in range(hops):
        nxt=set()
        for u,v in edge_list:
            if u in front:nxt.add(v)
            if v in front:nxt.add(u)
        front=nxt-seen;seen|=front
    pool=candidate_pairs(len(data.x),seen);budget=int(flips)
    if budget>len(pool):raise ValueError('Budget exceeds eligible pairs')
    if method=='greedy' and len(pool)>max_candidates:raise ValueError('Candidate space too large')
    old=model.training;model.eval();current=data;edits=[];rng=random.Random(seed)
    try:
        with torch.no_grad():
            before=float(torch.sigmoid(model(current)[target]).item())
            for _ in range(budget):
                if method=='random':pair=rng.choice(pool)
                else:
                    highest=-math.inf;pair=None
                    for p in pool:
                        cand=with_node_edges(current,edit_edges(current.edges,p,len(data.x),current.edges.device))
                        val=_objective(model(cand)[target],int(data.y[target]))
                        if val>highest+1e-12:highest=val;pair=p
                current=with_node_edges(current,edit_edges(current.edges,pair,len(data.x),current.edges.device))
                pool.remove(pair);edits.append(pair)
            after=float(torch.sigmoid(model(current)[target]).item())
    finally:model.train(old)
    assert len(canonical_pairs(data.edges)^canonical_pairs(current.edges))==budget
    y=int(data.y[target])
    return current,{'method':method,'target':target,'budget':budget,'eligible_nodes':len(seen),'edits':edits,
        'p_before':before,'p_after':after,'label':y,'pred_before':int(before>=.5),'pred_after':int(after>=.5),
        'delta_abs':abs(after-before),'attack_success':bool((after>=.5)!=(y==1)),
        'initial_correct':bool((before>=.5)==(y==1)),
        'label_preservation':'not verified; target label carried unchanged for evaluation only'}


def summarize(records):
    """Attack success on initially correct predictions; no misleading denominator."""
    if not records:return {'n':0,'initial_correct_n':0,'attack_success_rate_on_initially_correct':None,'mean_abs_probability_drift':None}
    eligible=[r for r in records if r['initial_correct']]
    return {'n':len(records),'initial_correct_n':len(eligible),
            'attack_success_rate_on_initially_correct':sum(r['attack_success'] for r in eligible)/len(eligible) if eligible else None,
            'mean_abs_probability_drift':float(np.mean([r['delta_abs'] for r in records]))}
