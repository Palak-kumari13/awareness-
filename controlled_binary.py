"""Matched, pre-registered binary node comparison. Never tune hyperparameters on test.
No guarantee of fairness improvement; fallback is part of the experiment, not post-hoc cherry-picking.
"""
import argparse,json
from vgr.node import synthetic_node_data
from vgr.guardrails import select_nonregressing

def run(seed,epochs):
    data=synthetic_node_data(seed,n=100)
    configs=[{'fair_weight':0.,'robust_weight':0.},
             {'fair_weight':0.03,'robust_weight':0.},
             {'fair_weight':0.1,'robust_weight':0.},
             {'fair_weight':0.03,'robust_weight':0.05},
             {'fair_weight':0.,'robust_weight':0.05}]
    return select_nonregressing(data,configs,seed=seed,epochs=epochs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--epochs',type=int,default=60);p.add_argument('--out',default='controlled_binary_results.json');a=p.parse_args()
    results=[run(seed,a.epochs) for seed in a.seeds]
    with open(a.out,'w') as f:json.dump({'protocol':'validation-only fairness selection; unchanged test splits; fixed configuration grid','results':results},f,indent=2)
    for r in results:
        b=r['baseline_test'];t=r['selected_test']
        print('seed',r['seed'],'selected',r['selected_index'],'baseline BA',b['balanced_accuracy'],'new BA',t['balanced_accuracy'],'baseline DP',b['dp_gap'],'new DP',t['dp_gap'],'test_nonregression',r['test_nonregression_observed'])
if __name__=='__main__':main()
