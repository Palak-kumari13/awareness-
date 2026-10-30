"""Controlled AngleFair ablation; no test-time hyperparameter selection."""
import argparse,json, numpy as np
from vgr.node import synthetic_node_data
from vgr.integrated import train_selected_node,report_node

def run(seed,epochs,weights):
    d=synthetic_node_data(seed,n=100)
    results=[]
    for weight in weights:
        model,hist=train_selected_node(d,seed=seed,epochs=epochs,angle_weight=weight)
        reports=report_node(model,d)
        results.append({'angle_weight':weight,'train':reports['train'],'val':reports['val'],'test':reports['test']})
    return {'seed':seed,'results':results}

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--epochs',type=int,default=60);p.add_argument('--weights',type=float,nargs='+',default=[0.,.001,.01]);p.add_argument('--out',default='anglefair_results.json');a=p.parse_args()
    res=[run(seed,a.epochs,a.weights) for seed in a.seeds]
    with open(a.out,'w') as f:json.dump({'protocol':'same seed/split; no tuning on test; fixed threshold 0.5; experimental angle loss', 'runs':res},f,indent=2)
    for r in res:
        for t in r['results']:
            q=t['test'];print('seed',r['seed'],'weight',t['angle_weight'],'test BA',round(q.get('balanced_accuracy',float('nan')),4),'DP',round(q.get('dp_gap',float('nan')),4),'AUROC',q.get('auc'))
if __name__=='__main__':main()
