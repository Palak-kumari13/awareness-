import unittest
import torch
from vgr.core import GNN,Graph
from vgr.node import NodeGNN,synthetic_node_data
from vgr.phase6 import graph_attack,node_attack,attack_budget,canonical_pairs,summarize

class Phase6Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.g=Graph(torch.randn(6,3),torch.tensor([[0,1],[1,2],[2,3],[3,4],[4,5]],dtype=torch.long),1)
        self.m=GNN(3,hidden=8,selector=True)

    def test_zero_flip_identity(self):
        attacked,r=graph_attack(self.m,self.g,'greedy',flips=0)
        self.assertTrue(torch.equal(attacked.edges,self.g.edges))
        self.assertAlmostEqual(r['p_before'],r['p_after'])

    def test_random_flip_exact_and_features(self):
        attacked,r=graph_attack(self.m,self.g,'random',flips=3,seed=4)
        self.assertEqual(len(canonical_pairs(attacked.edges)^canonical_pairs(self.g.edges)),3)
        self.assertTrue(torch.equal(attacked.x,self.g.x))
        self.assertEqual(len(set(map(tuple,r['edits']))),3)

    def test_greedy_flip_exact(self):
        attacked,r=graph_attack(self.m,self.g,'greedy',flips=2)
        self.assertEqual(len(canonical_pairs(attacked.edges)^canonical_pairs(self.g.edges)),2)
        self.assertEqual(r['budget'],2)

    def test_reproducible_random(self):
        a,x=graph_attack(self.m,self.g,'random',flips=2,seed=3)
        b,y=graph_attack(self.m,self.g,'random',flips=2,seed=3)
        self.assertEqual(x['edits'],y['edits'])
        self.assertTrue(torch.equal(a.edges,b.edges))

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):graph_attack(self.m,self.g,flips=16)
        with self.assertRaises(ValueError):attack_budget(5,rate=1.2)

    def test_model_mode_restored(self):
        self.m.train(True)
        graph_attack(self.m,self.g,flips=1)
        self.assertTrue(self.m.training)

    def test_node_attack_indices_unchanged(self):
        d=synthetic_node_data(seed=2,n=60)
        m=NodeGNN(d.x.shape[1],hidden=8)
        t=int(d.test_idx[1]);z,r=node_attack(m,d,t,'greedy',flips=1,hops=5)
        self.assertEqual(len(canonical_pairs(z.edges)^canonical_pairs(d.edges)),1)
        self.assertTrue(torch.equal(z.train_idx,d.train_idx))
        self.assertTrue(torch.equal(z.x,d.x))
        self.assertEqual(r['target'],t)

    def test_success_denominator(self):
        r=summarize([{'initial_correct':False,'attack_success':True,'delta_abs':.3}])
        self.assertIsNone(r['attack_success_rate_on_initially_correct'])

if __name__=='__main__':unittest.main()
