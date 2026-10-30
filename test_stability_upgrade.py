import unittest
import numpy as np
import torch
from vgr.core import GNN
from vgr.data import synthetic,split
from vgr.trainlib import train
from vgr.decision import choose_threshold,classification_audit

class StabilityUpgradeTests(unittest.TestCase):
    def test_val_threshold_uses_two_classes(self):
        x=choose_threshold([0,0,1,1],[.3,.4,.6,.7]);self.assertTrue(x['class_recall_constraint_feasible'])
    def test_new_checkpoint_rule_is_recorded(self):
        graphs=synthetic(n_graphs=40,seed=8);tr,va,te=split(graphs,8)
        m,h=train(GNN(1,hidden=8,selector=False),tr,va,epochs=2,seed=8)
        self.assertTrue(all('validation_auroc' in row for row in h))
    def test_class_collapse_still_rejected(self):
        d=classification_audit([0,1,0,1],[.8,.8,.8,.8]);self.assertFalse(d['gate_pass'])
    def test_selected_edges_are_hard_and_budgeted(self):
        g=synthetic(n_graphs=2,seed=4)[0];m=GNN(1,hidden=8,budget=.5)
        w,h,s=m.select(g)
        self.assertTrue(torch.equal(w.detach(),h));self.assertEqual(int(h.sum()),max(1,round(len(g.edges)*.5)))
