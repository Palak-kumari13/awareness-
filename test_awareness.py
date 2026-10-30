import unittest
import torch
from vgr.awareness import structural_features
from awareness_experiment import social_data,AwareSelectedGNN
class AwarenessTests(unittest.TestCase):
 def test_variants_dimensions(self):
  g=social_data(2,15)[0]
  for mode,n in [('topology',6),('motif',7),('spectral',7),('hybrid',11)]:
   node,edge,adj=structural_features(g.edges,len(g.x),mode)
   self.assertEqual(edge.shape[1],n)
   self.assertEqual(adj.shape,(len(g.x),len(g.x)))
 def test_edge_order_invariance(self):
  g=social_data(2,19)[0];m=AwareSelectedGNN('topology');m.eval()
  from vgr.core import Graph
  with torch.no_grad():
   a=float(m(g));b=float(m(Graph(g.x,g.edges.flip(0),g.y,g.sensitive)))
  self.assertAlmostEqual(a,b,places=5)
 def test_selector_not_constant(self):
  g=social_data(2,17)[0];m=AwareSelectedGNN('motif');_,mask,scores,_=m(g,True)
  self.assertGreater(float(scores.std()),0)
  self.assertEqual(int(mask.sum()),round(.6*len(g.edges)))
 def test_training_gradient(self):
  g=social_data(2,29)[0];m=AwareSelectedGNN('hybrid');m(g).backward()
  self.assertTrue(m.edge_score[0].weight.grad is not None)
  self.assertTrue(torch.isfinite(m.edge_score[0].weight.grad).all())
if __name__=='__main__':unittest.main()
