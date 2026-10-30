import unittest
import torch
from vgr.research_gates import select_checkpoint,explanation_evidence
from vgr.frme_controls import edge_deletion_effect,joint_edge_interaction
from vgr.core import Graph,GNN

class ResearchMerge(unittest.TestCase):
 def test_guard_missing_metrics_fails_closed(self):
  b={'auroc':.8,'balanced_accuracy':.75,'collapsed':False,'dp_gap':.1,'eo_gap':.1,'prediction_drift':.01}
  res=select_checkpoint([b,{**b,'dp_gap':None,'auroc':.85}])
  self.assertEqual(res['chosen_index'],0)
 def test_explanations_do_not_certify_on_spread_only(self):
  self.assertFalse(explanation_evidence({'score_std':.3})['passed'])
 def test_explanation_matched_controls(self):
  row={'score_std':.2,'selected_fidelity':.01,'random_fidelity':.1,'degree_fidelity':.08}
  self.assertTrue(explanation_evidence(row)['passed'])
  self.assertFalse(explanation_evidence({**row,'selected_fidelity':.09})['passed'])
 def test_interventions_do_not_mutate_original(self):
  g=Graph(torch.ones(4,1),torch.tensor([[0,1],[1,2],[2,3]],dtype=torch.long),1)
  model=GNN(1,hidden=8,selector=False)
  before=g.edges.clone();result=edge_deletion_effect(model,g,[0]);self.assertTrue(torch.equal(before,g.edges))
  self.assertTrue(0<=result['after_p1']<=1)
  pair=joint_edge_interaction(model,g,0,1)
  self.assertAlmostEqual(pair['interaction'],pair['joint']-pair['first']-pair['second'],places=6)
 def test_bad_edge_rejected(self):
  g=Graph(torch.ones(3,1),torch.tensor([[0,1]],dtype=torch.long),0)
  with self.assertRaises(IndexError):edge_deletion_effect(GNN(1,hidden=8),g,[99])

if __name__=='__main__':unittest.main()
