"""Standalone Phase 6 runner; leaves earlier launchers unchanged."""
import argparse,json
from pathlib import Path
import torch
from vgr.core import GNN
from vgr.data import synthetic,load_tu,split
from vgr.trainlib import seed_all,train
from vgr.node import load_node_npz,synthetic_node_data
from vgr.fairness import train_node
from vgr.phase6 import graph_attack,node_attack,summarize,attack_budget

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['graph','node'],default='graph')
    p.add_argument('--dataset',default='synthetic')
    p.add_argument('--npz',default=None)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--budget',type=float,default=.5,help='fraction of edges kept by graph selector')
    p.add_argument('--flip-budget',type=int,default=1)
    p.add_argument('--rate',type=float,default=None,help='alternative flip budget as fraction of original edges')
    p.add_argument('--examples',type=int,default=5)
    p.add_argument('--max-candidates',type=int,default=5000)
    p.add_argument('--hops',type=int,default=2)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',default='outputs/phase6')
    a=p.parse_args();seed_all(a.seed)
    if a.epochs<1 or a.examples<1:raise ValueError('epochs and examples must be positive')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    if a.task=='graph':
        graphs=synthetic(seed=a.seed) if a.dataset=='synthetic' else load_tu(a.dataset)
        tr,va,te=split(graphs,a.seed)
        model=GNN(tr[0].x.shape[1],hidden=a.hidden,selector=True,budget=a.budget)
        model,_=train(model,tr,va,epochs=a.epochs,seed=a.seed,device=a.device)
        for i,g in enumerate(te[:a.examples]):
            g=g.to(a.device)
            flips=attack_budget(len(g.edges),rate=a.rate) if a.rate is not None else a.flip_budget
            for method in ('random','greedy'):
                _,r=graph_attack(model,g,method=method,flips=flips,seed=a.seed+i,max_candidates=a.max_candidates)
                rows.append({'example':i,**r})
    else:
        if a.npz:data=load_node_npz(a.npz,seed=a.seed)
        elif a.dataset=='synthetic-node':data=synthetic_node_data(a.seed)
        else:raise ValueError('Node data require --npz or synthetic-node')
        model,_,_=train_node(data,epochs=a.epochs,seed=a.seed,hidden=a.hidden,device=a.device)
        data=data.to(a.device)
        for i,target in enumerate([int(t) for t in data.test_idx.tolist() if bool(((data.edges == int(t)).any(dim=1)).any())][:a.examples]):
            flips=attack_budget(len(data.edges),rate=a.rate) if a.rate is not None else a.flip_budget
            for method in ('random','greedy'):
                _,r=node_attack(model,data,target,method=method,flips=flips,seed=a.seed+i,hops=a.hops,max_candidates=a.max_candidates)
                rows.append({'example':i,**r})
    result={'task':a.task,'dataset':a.dataset,'seed':a.seed,'threat_model':'frozen-model test-time evasion; undirected edge flips; labels/features held fixed; semantic label preservation not guaranteed',
            'attacks':rows,'summary':{m:summarize([r for r in rows if r['method']==m]) for m in ('random','greedy')}}
    (out/'phase6_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(f'Phase 6 results: {out / "phase6_results.json"}')
if __name__=='__main__':main()
