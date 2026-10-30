"""Independent empirical verification: same frozen model for all graph interventions."""
import random
import numpy as np
import torch
from .core import Graph,intervention,selected_graph

@torch.no_grad()
def prob(model,g):
    return float(torch.sigmoid(model(g)).cpu())


def rank_edges(model,g,method='deletion'):
    """Per-edge original-class probability drop. Scores are *model dependence*.
    For symmetric graph labels use absolute class 1/0 confidence as appropriate.
    """
    model.eval(); base=prob(model,g); sign=1 if base>=.5 else -1
    if method=='deletion':
        return np.array([sign*(base-prob(model,intervention(g,[i]))) for i in range(len(g.edges))])
    if method=='degree':
        d=np.bincount(g.edges.flatten().cpu().numpy(),minlength=len(g.x))
        return np.array([d[int(u)]+d[int(v)] for u,v in g.edges])
    if method=='random':
        return np.random.default_rng(0).random(len(g.edges))
    if method=='selector':
        with torch.no_grad(): return model(g,return_details=True)['scores'].cpu().numpy()
    if method=='saliency':
        # ST surrogate derivatives of soft edge weights (evaluated at full weight)
        model.zero_grad(set_to_none=True)
        weights=torch.ones(len(g.edges),requires_grad=True,device=g.x.device)
        with torch.enable_grad():
            logit=model(g,edge_weights=weights)
            grad=torch.autograd.grad(logit,weights)[0].detach().abs().cpu().numpy()
        return grad
    if method=='pagerank':
        import networkx as nx
        net=nx.Graph();net.add_nodes_from(range(len(g.x)));net.add_edges_from(g.edges.cpu().tolist())
        pr=nx.pagerank(net)
        return np.array([pr[int(u)]+pr[int(v)] for u,v in g.edges])
    raise ValueError(method)

@torch.no_grad()
def audit_graph(model,g,method='selector',fraction=.5,seed=0,control_reps=16):
    """Keep/drop on frozen *selected-graph* predictor. Re-evaluates model on actual
    edge deletions with its selector disabled temporarily to avoid reselecting the
    intervention; tests fidelity of the original selected edge set.
    """
    model.eval()
    original=model.selector_on
    model.selector_on=False
    try:
        full=prob(model,g); confsign=1 if full>=.5 else -1
        m=len(g.edges); k=min(m,max(1,round(m*fraction))) if m else 0
        if method=='selector' and original:
            model.selector_on=True
            scores=rank_edges(model,g,'selector')
            model.selector_on=False
        else: scores=rank_edges(model,g,method)
        canonical=g.edges[:,0].cpu().numpy()*len(g.x)+g.edges[:,1].cpu().numpy()
        order=np.lexsort((canonical,-scores)); selected=set(order[:k].tolist())
        def assess(index_set):
            keep=[i for i in range(m) if i in index_set]
            drop=[i for i in range(m) if i not in index_set]
            pkeep=prob(model,intervention(g,drop))
            pdrop=prob(model,intervention(g,keep))
            return dict(full=full,keep=pkeep,drop=pdrop,
                        suff_error=abs(full-pkeep), necessity=confsign*(full-pdrop),
                        flip_after_drop=int((full>=.5)!=(pdrop>=.5)))
        result=assess(selected)
        rng=random.Random(seed)
        controls=[assess(set(rng.sample(range(m),k))) for _ in range(control_reps)] if m else []
        result.update({'method':method,'edges':m,'selected':k,'selected_edge_ids':sorted(selected),
                       'score_std':float(np.std(scores)) if m else 0,
                       'random_suff_mean':float(np.mean([x['suff_error'] for x in controls])) if controls else float('nan'),
                       'random_nec_mean':float(np.mean([x['necessity'] for x in controls])) if controls else float('nan')})
        if g.motif is not None and k:
            truth=set(torch.where(g.motif)[0].tolist())
            result['motif_precision']=len(truth & selected)/k
            result['motif_recall']=len(truth & selected)/max(len(truth),1)
        return result
    finally:
        model.selector_on=original


def permutation_test(model,g):
    model.eval(); rng=torch.Generator().manual_seed(123)
    order=torch.randperm(len(g.edges),generator=rng)
    shuffled=Graph(g.x,g.edges[order],g.y,g.sensitive)
    with torch.no_grad():
        a=model(g,return_details=True); b=model(shuffled,return_details=True)
    # compare selected physical edge sets, not row positions
    def ids(graph,mask):
        return {(int(u),int(v)) for (u,v),chosen in zip(graph.edges.tolist(),mask.tolist()) if chosen}
    return {'logit_delta':float(torch.abs(a['logit']-b['logit'])),
            'selection_equal':ids(g,a['mask'])==ids(shuffled,b['mask'])}
