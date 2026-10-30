"""Fair controlled diagnostic; runs without PyTorch Geometric."""
import argparse,json,copy
from pathlib import Path
from vgr.core import GNN
from vgr.trainlib import train,metrics,seed_all
from vgr.selector_repair import frozen_split,train_selector_variant,explain_precision
from vgr.decision import classification_audit
import torch

def run(seeds,epochs=60,graphs=180,hidden=32,budget=.5,lr=.001,layers=2,method_names=('joint','alternate','teacher')):
    records=[]
    for seed in seeds:
        trainset,valset,testset=frozen_split(seed,graphs)
        seed_all(seed)
        ref=GNN(1,hidden=hidden,layers=layers,selector=False)
        ref,_=train(ref,trainset,valset,epochs=epochs,lr=lr,seed=seed,patience=15)
        records.append({'seed':seed,'method':'full_graph','test':metrics(ref,testset)})
        for method in method_names:
            seed_all(seed)
            model=GNN(1,hidden=hidden,layers=layers,selector=True,budget=budget)
            model,hist=train_selector_variant(model,trainset,valset,method=method,reference=ref if method=='teacher' else None,
                                             epochs=epochs,seed=seed,lr=lr)
            row={'seed':seed,'method':method,'test':metrics(model,testset),
                 'explanation':explain_precision(model,testset,seed=seed),'epochs_run':len(hist)}
            records.append(row)
            print(seed,method,round(row['test']['auroc'],3),round(row['test']['balanced_accuracy'],3),row['explanation'],flush=True)
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4]);p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--graphs',type=int,default=180);p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--budget',type=float,default=.5);p.add_argument('--layers',type=int,choices=[2,3],default=2);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--out',default='outputs/selector_repair.json');args=p.parse_args()
    result=run(args.seeds,args.epochs,args.graphs,args.hidden,args.budget,args.lr,args.layers)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False))
