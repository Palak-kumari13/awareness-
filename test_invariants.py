import unittest
import torch
from vgr.core import GNN,Graph,edge_adjacency,intervention
from vgr.data import synthetic,split
from vgr.audit import audit_graph,permutation_test
from vgr.trainlib import metrics

class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(12)
        cls.g=synthetic(n_graphs=10,seed=12)[0]
        cls.model=GNN(1,hidden=8,budget=.4)

    def test_exact_hard_budget(self):
        p=self.model(self.g,return_details=True)
        self.assertEqual(int(p['mask'].sum()),max(1,round(.4*len(self.g.edges))))
        self.assertTrue(bool(((p['mask']==0)|(p['mask']==1)).all()))

    def test_selector_gradient(self):
        self.model.train()
        self.model.zero_grad()
        self.model(self.g).backward()
        grads=[p.grad for p in self.model.scores.parameters()]
        self.assertTrue(all(g is not None for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)

    def test_edge_permutation(self):
        result=permutation_test(self.model,self.g)
        self.assertTrue(result['selection_equal'])
        self.assertLess(result['logit_delta'],1e-5)

    def test_true_deletion(self):
        a=edge_adjacency(self.g)
        sub=intervention(self.g,[0]);b=edge_adjacency(sub)
        u,v=self.g.edges[0].tolist()
        self.assertEqual(float(b[u,v]),0.)
        self.assertEqual(float(a[u,v]),1.)
        self.assertEqual(len(sub.x),len(self.g.x))

    def test_audit_does_not_change_selector(self):
        _=audit_graph(self.model,self.g,control_reps=3)
        self.assertTrue(self.model.selector_on)

    def test_split_disjoint_sizes(self):
        data=synthetic(100,seed=5)
        parts=split(data,5)
        self.assertEqual(sum(map(len,parts)),100)

if __name__=='__main__':unittest.main()
