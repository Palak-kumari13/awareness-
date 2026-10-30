"""Run failure-aware post-training audit using only validation to select threshold.

Runs on the existing graph V0/V1/V2 and node V4/V5 training routines.
No claim of verified fairness or robustness is made.
"""
import argparse,json
from pathlib import Path
from vgr.decision import (choose_threshold,classification_audit,
                          prediction_probabilities,node_prediction_probabilities)
from vgr.trainlib import train,seed_all
from vgr.node import synthetic_node_data,load_node_npz
from vgr.fairness import train_node, fairness_gaps
from vgr.data import synthetic,load_tu,split
from vgr.core import GNN


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['graph','node'],required=True)
    p.add_argument('--dataset',default='synthetic')
    p.add_argument('--npz',default=None)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--threshold-min-recall',type=float,default=.1)
    p.add_argument('--out',default='outputs/decision_audit')
    args=p.parse_args();seed_all(args.seed)
    if args.task=='graph':
        dataset=synthetic(seed=args.seed) if args.dataset=='synthetic' else load_tu(args.dataset)
        tr,va,te=split(dataset,seed=args.seed)
        model=GNN(tr[0].x.size(1),hidden=32,selector=True,budget=.5)
        model,_=train(model,tr,va,epochs=args.epochs,seed=args.seed)
        yv,pv=prediction_probabilities(model,va)
        yt,pt=prediction_probabilities(model,te)
        fair=None
    else:
        if args.npz:data=load_node_npz(args.npz,seed=args.seed)
        elif args.dataset=='synthetic-node':data=synthetic_node_data(args.seed)
        else:raise ValueError('Specify --npz or --dataset synthetic-node')
        model,_,status=train_node(data,epochs=args.epochs,seed=args.seed)
        yv,pv=node_prediction_probabilities(model,data,'val')
        yt,pt=node_prediction_probabilities(model,data,'test')
        fair=data.sensitive[data.test_idx].numpy()
    selection=choose_threshold(yv,pv,min_class_recall=args.threshold_min_recall)
    report={'validation_threshold_selection':selection,
            'test_at_default_0_5':classification_audit(yt,pt,threshold=.5),
            'test_at_validation_threshold':classification_audit(yt,pt,threshold=selection['threshold'])}
    if fair is not None:
        report['fairness_at_validation_threshold']=fairness_gaps(yt,pt,fair,selection['threshold'])
        report['fairness_caution']='Zero demographic parity gap under constant predictions is not evidence of useful fairness.'
    report['scientific_warning']='This is a classifier decision audit, not verification of explanation fidelity or robustness.'
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'decision_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
