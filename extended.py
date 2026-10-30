"""Experiment entry point for phases 2-5. Never evaluates real data until provided."""
import argparse,csv,json
from pathlib import Path
import torch
from vgr.data import synthetic,load_tu,split
from vgr.core import GNN
from vgr.phase2 import compare_baselines
from vgr.phase3 import group_removal_audit,concept_interventions,motif_gate,shuffle_label_audit
from vgr.node import load_node_npz,synthetic_node_data,ego_graph
from vgr.fairness import fairness_sweep,node_metrics
from vgr.trainlib import train,seed_all,metrics

def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda o: None if isinstance(o,float) and not torch.isfinite(torch.tensor(o)) else str(o)))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['2','3','4','5'],required=True)
    p.add_argument('--dataset',default='synthetic')
    p.add_argument('--npz',default=None,help='Node datasets: explicit NPZ schema in README')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--budget',type=float,default=.5)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',default='outputs/extended')
    args=p.parse_args();seed_all(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.phase in ('2','3'):
        graphs=synthetic(seed=args.seed) if args.dataset=='synthetic' else load_tu(args.dataset)
        tr,va,te=split(graphs,args.seed)
        if args.phase=='2':
            result=compare_baselines(tr,va,te,args.epochs,args.hidden,args.seed,args.device)
        else:
            model=GNN(tr[0].x.size(1),hidden=args.hidden,selector=True,budget=args.budget)
            model,_=train(model,tr,va,epochs=args.epochs,seed=args.seed,device=args.device)
            model.eval()
            result={'predictor_test':metrics(model,te,args.device),
                    'group_removal':group_removal_audit(model,te[0].to(args.device)),
                    'concept_interventions':concept_interventions(model,te[0].to(args.device)),
                    'shuffle_label_control':shuffle_label_audit(tr,va,te,args.seed,args.epochs,args.hidden,args.device)}
            if all(g.motif is not None for g in te):
                result['planted_motif']=motif_gate(model,[g.to(args.device) for g in te],budget=args.budget,seed=args.seed)
            torch.save({'state_dict':model.cpu().state_dict(),'input_dim':tr[0].x.size(1),
                        'hidden':args.hidden,'budget':args.budget},out/'phase3_model.pt')
    else:
        if args.npz:data=load_node_npz(args.npz,seed=args.seed)
        elif args.dataset=='synthetic-node':data=synthetic_node_data(args.seed)
        else:raise ValueError('Provide --npz path for real node datasets; no unverified automatic downloads')
        if args.phase=='4':
            from vgr.fairness import train_node
            model,hist,status=train_node(data,epochs=args.epochs,seed=args.seed,hidden=args.hidden,device=args.device)
            ego,target,nodes=ego_graph(data,int(data.test_idx[0]),hops=2)
            result={'status':status,'validation':node_metrics(model,data,'val',args.device),
                    'test':node_metrics(model,data,'test',args.device),
                    'ego_example':{'target_global_id':int(data.test_idx[0]),'target_local_id':target,
                                   'node_ids':nodes,'edges':int(len(ego.edges))}}
            torch.save({'state_dict':model.cpu().state_dict(),'input_dim':data.x.size(1),
                        'hidden':args.hidden},out/'phase4_node_model.pt')
        else:
            runs,models=fairness_sweep(data,weights=(0.,.03,.1,.3),epochs=args.epochs,
                                       seed=args.seed,hidden=args.hidden,device=args.device)
            result={'runs':runs,'note':'Utility floor is validation accuracy of baseline - 0.02; threshold fixed at 0.5'}
            for weight,model in models.items():
                torch.save({'state_dict':model.cpu().state_dict(),'fair_weight':weight},out/f'phase5_fair_{weight}.pt')
    write_json(out/'results.json',result)
    print(f'Completed phase {args.phase}; results: {out / "results.json"}')
if __name__=='__main__':main()
