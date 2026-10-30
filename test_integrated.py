import unittest,torch
from vgr.integrated import planted_multilabel,MultiLabelGNN,SelectedNodeGNN,train_selected_node
from vgr.node import synthetic_node_data
class IntegratedTests(unittest.TestCase):
 def test_multilabel_independent(self):
  samples=planted_multilabel(24,1)
  self.assertEqual(set(tuple(y.tolist()) for _,y in samples),{(0.,0.),(0.,1.),(1.,0.),(1.,1.)})
 def test_multilabel_forward(self):
  g,y=planted_multilabel(4,0)[0]
  m=MultiLabelGNN(1,labels=2,selector=True)
  self.assertEqual(tuple(m(g).shape),(2,))
 def test_node_selector_grad(self):
  d=synthetic_node_data(0,40);m=SelectedNodeGNN(4)
  m(d)[d.train_idx].sum().backward()
  grad=sum(p.grad.abs().sum().item() for p in m.scorer.parameters() if p.grad is not None)
  self.assertGreater(grad,0)
 def test_node_selection_budget(self):
  d=synthetic_node_data(0,40);m=SelectedNodeGNN(4,budget=.6)
  _,v=m(d,True)
  self.assertEqual(int(v['hard_selected'].sum()),round(.6*len(d.edges)))
if __name__=='__main__':unittest.main()
