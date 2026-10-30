import unittest,tempfile,os
import numpy as np
import torch
from vgr.data import synthetic,split
from vgr.core import GNN
from vgr.phase2 import FullGraphBackbone,degree_preserving_rewire,majority_result
from vgr.phase3 import group_removal_audit,concept_interventions,motif_gate
from vgr.node import synthetic_node_data,ego_graph,load_node_npz,NodeGNN
from vgr.fairness import fairness_gaps,train_node,node_metrics,fairness_sweep

class Phase2345Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs=synthetic(n_graphs=24,seed=14)
    def test_baseline_models_forward_backward(self):
        for kind in ['mlp','gin','gcn','sage']:
            m=FullGraphBackbone(1,hidden=8,kind=kind)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(m(self.graphs[0]),torch.tensor(float(self.graphs[0].y)))
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in m.parameters()))
    def test_degree_preserving(self):
        g=self.graphs[0];h=degree_preserving_rewire(g,seed=9)
        def degrees(z):
            return torch.bincount(z.edges.reshape(-1),minlength=len(z.x))
        self.assertTrue(torch.equal(degrees(g),degrees(h)))
        self.assertTrue(torch.equal(g.x,h.x))
    def test_group_and_concepts(self):
        g=self.graphs[0];m=GNN(1,hidden=8,selector=True)
        res=group_removal_audit(m,g,group_size=2,top_n=4,random_reps=3)
        self.assertEqual(len(res['groups']),6)
        cs=concept_interventions(m,g)
        self.assertEqual(len(cs),3)
    def test_motif_gate(self):
        m=GNN(1,hidden=8)
        r=motif_gate(m,self.graphs[:4],budget=.5)
        self.assertTrue(0<=r['precision']<=1)
    def test_node_ego_and_forward(self):
        d=synthetic_node_data(n=60,seed=6)
        g,target,nodes=ego_graph(d,int(d.test_idx[0]),2)
        self.assertEqual(nodes[target],int(d.test_idx[0]))
        m=NodeGNN(4,hidden=8)
        self.assertEqual(tuple(m(d).shape),(60,))
    def test_node_npz_roundtrip(self):
        d=synthetic_node_data(n=60,seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            path=os.path.join(tmp,'node.npz')
            np.savez(path,x=d.x.numpy(),edges=d.edges.numpy(),y=d.y.numpy(),
                     sensitive=d.sensitive.numpy(),train_idx=d.train_idx.numpy(),
                     val_idx=d.val_idx.numpy(),test_idx=d.test_idx.numpy())
            loaded=load_node_npz(path)
            self.assertTrue(torch.equal(loaded.edges,d.edges))
    def test_fairness_gaps_and_undefined_strata(self):
        x=fairness_gaps([0,1,0,1],[.1,.9,.2,.8],[0,0,1,1])
        self.assertEqual(x['dp_gap'],0)
        y=fairness_gaps([0,0,1],[.2,.3,.8],[0,0,1])
        self.assertIsNone(y['eo_gap'])
    def test_fairness_floor_flag(self):
        d=synthetic_node_data(seed=4,n=60)
        m,h,s=train_node(d,epochs=2,hidden=8,utility_floor=1.01)
        self.assertFalse(s['floor_feasible'])
        met=node_metrics(m,d,'test')
        self.assertIn('dp_gap',met)
    def test_fairness_train_and_sweep(self):
        d=synthetic_node_data(seed=3,n=60)
        m,h,s=train_node(d,epochs=2,hidden=8,fair_weight=.1)
        self.assertEqual(len(h),2)
        result,_=fairness_sweep(d,weights=(0.,.1),epochs=2,hidden=8)
        self.assertEqual(len(result),2)

if __name__=='__main__':unittest.main()
