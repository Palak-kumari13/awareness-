"""Integrated branch experiments, separate from historical V0-V6 results."""
import argparse,json,random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from vgr.integrated import planted_multilabel,train_graph_multilabel,evaluate_multilabel,train_selected_node,report_node
from vgr.node import synthetic_node_data
from vgr.fairness import fairness_gaps

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['multilabel','node'],default='multilabel')
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--graphs',type=int,default=120);p.add_argument('--fair-weight',type=float,default=0.05)
    p.add_argument('--robust-weight',type=float,default=.1);p.add_argument('--output',default='integrated_results.json')
    a=p.parse_args()
    if a.task=='multilabel':
        items=planted_multilabel(a.graphs,a.seed)
        ids=np.arange(len(items));strata=np.array([int(y[0]+2*y[1]) for _,y in items]);
        tr,temp=train_test_split(ids,test_size=.4,random_state=a.seed,stratify=strata)
        va,te=train_test_split(temp,test_size=.5,random_state=a.seed,stratify=strata[temp])
        train=[items[i] for i in tr];valid=[items[i] for i in va];test=[items[i] for i in te]
        model,hist=train_graph_multilabel(train,seed=a.seed,epochs=a.epochs)
        # Fixed thresholds to avoid small validation-set overfit; can add calibrated threshold workflow later
        result={'task':a.task,'train':evaluate_multilabel(model,train),
                'val':evaluate_multilabel(model,valid),'test':evaluate_multilabel(model,test),
                'train_loss_last':hist[-1]}
    else:
        d=synthetic_node_data(a.seed,n=100)
        model,hist=train_selected_node(d,seed=a.seed,epochs=a.epochs,fair_weight=a.fair_weight,
                                      robust_weight=a.robust_weight)
        result={'task':a.task,'metrics':report_node(model,d),'train_loss_last':hist[-1]}
    with open(a.output,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
