import argparse,json,csv
from pathlib import Path
from vgr.core import GNN
from vgr.data import synthetic,load_tu,split
from vgr.trainlib import seed_all,train,metrics
from vgr.audit import audit_graph,permutation_test
from vgr.decision import choose_threshold,classification_audit,prediction_probabilities

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='synthetic',help='synthetic, IMDB-BINARY, MUTAG, PROTEINS, NCI1')
    p.add_argument('--stage',default='V1',choices=['V0','V1','V2','V3','V4','V5'])
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--hidden',type=int,default=32);p.add_argument('--budget',type=float,default=.5)
    p.add_argument('--out',default='outputs/run');p.add_argument('--device',default='cpu')
    args=p.parse_args();seed_all(args.seed)
    dataset=synthetic(seed=args.seed) if args.dataset.lower()=='synthetic' else load_tu(args.dataset)
    tr,va,te=split(dataset,seed=args.seed)
    model=GNN(tr[0].x.size(1),hidden=args.hidden,selector=args.stage!='V0',budget=args.budget,concepts_on=True)
    model,history=train(model,tr,va,epochs=args.epochs,seed=args.seed,device=args.device,
                        fair_weight=.1 if args.stage in ('V3','V5') and all(g.sensitive is not None for g in tr) else 0,
                        robust_weight=.5 if args.stage in ('V4','V5') else 0)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.save({'state_dict':model.state_dict(),'input_dim':tr[0].x.size(1),'stage':args.stage,
                'hidden':args.hidden,'budget':args.budget},out/'checkpoint.pt')
    results={'train':metrics(model,tr,args.device),'validation':metrics(model,va,args.device),
             'test':metrics(model,te,args.device),
             'permutation':permutation_test(model,te[0].to(args.device)),
             'config':vars(args)}
    (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=True))
    (out/'history.json').write_text(json.dumps(history,indent=2))
    if args.stage!='V0':
        rows=[]
        for i,g in enumerate(te[:min(20,len(te))]):
            for method in ('selector','deletion','degree','pagerank','saliency'):
                row=audit_graph(model,g.to(args.device),method=method,fraction=args.budget,seed=args.seed+i,control_reps=8)
                row['graph_id']=i
                rows.append(row)
        with (out/'audit.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader();writer.writerows(rows)
    yv,pv=prediction_probabilities(model,va,args.device)
    yt,pt=prediction_probabilities(model,te,args.device)
    decision=choose_threshold(yv,pv)
    results['decision_audit']={'validation_selection':decision,
       'test_default':classification_audit(yt,pt,.5),
       'test_validation_threshold':classification_audit(yt,pt,decision['threshold'])}
    results['test']['collapsed_predictions']=results['decision_audit']['test_default']['collapsed_predictions']
    results['gate_pass']=bool(results['decision_audit']['test_validation_threshold']['gate_pass'])
    (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=True))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
