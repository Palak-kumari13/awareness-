"""Audit intrinsic selection on synthetic graph motifs and node perturbation sensitivity.
No true causal claims. Fixed model checkpoints, no test tuning.
"""
import json, random, numpy as np, torch
from vgr.integrated import planted_multilabel, train_graph_multilabel, train_selected_node
from vgr.node import synthetic_node_data,NodeData

def motif_audit(model,items):
    model.eval();q=[];random_scores=[]
    with torch.no_grad():
        for g,y in items:
            if not g.motif.any():continue
            d=model(g,return_details=True);mask=d['mask'].bool();truth=g.motif
            q.append(float((mask&truth).sum()/mask.sum().clamp(min=1)))
            random_scores.append(float(truth.float().mean()))
    return {'precision':float(np.mean(q)) if q else None,'random_expected_precision':float(np.mean(random_scores)) if q else None,'graphs_with_motif':len(q)}

def edit_stability(model,d,seed=20,trials=8):
    rng=random.Random(seed);model.eval()
    with torch.no_grad():
        base=model(d)[d.test_idx].sigmoid()
        vals=[]
        for t in range(trials):
            keep=torch.tensor([rng.random()>.05 for e in d.edges],dtype=torch.bool)
            changed=NodeData(d.x,d.edges[keep],d.y,d.sensitive,d.train_idx,d.val_idx,d.test_idx)
            vals.append(float((model(changed)[d.test_idx].sigmoid()-base).abs().mean()))
    return {'mean_absolute_probability_drift':float(np.mean(vals)),'trials':trials,'attack':'random 5% edge removal, not model-aware'}

if __name__=='__main__':
    samples=planted_multilabel(120,0);m,_=train_graph_multilabel(samples[:72],epochs=35)
    print(json.dumps({'motif':motif_audit(m,samples[72:])},indent=2))
    d=synthetic_node_data(0,100)
    for rw,fw in [(0.,0.),(.1,.05)]:
        m,_=train_selected_node(d,epochs=65,robust_weight=rw,fair_weight=fw)
        print(json.dumps({'robust_weight':rw,'fair_weight':fw,'random_edit_stability':edit_stability(m,d)},indent=2))
