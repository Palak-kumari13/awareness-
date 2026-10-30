import unittest
import torch
from vgr.anglefair import principal_angle_alignment,anglefair_training_step
from vgr.node import synthetic_node_data
from vgr.integrated import SelectedNodeGNN
class AngleFairTests(unittest.TestCase):
 def test_missing_groups_no_false_success(self):
  h=torch.randn(8,4,requires_grad=True); y=torch.zeros(8,dtype=torch.long);s=torch.zeros(8,dtype=torch.long)
  loss,active=principal_angle_alignment(h,y,s)
  self.assertEqual(active,0);self.assertEqual(float(loss),0.)
 def test_gradient(self):
  torch.manual_seed(3);h=torch.randn(24,5,requires_grad=True);y=torch.arange(24)%2;s=(torch.arange(24)//2)%2
  loss,active=principal_angle_alignment(h,y,s)
  self.assertEqual(active,2)
  loss.backward();self.assertTrue(torch.isfinite(h.grad).all());self.assertGreater(h.grad.abs().sum().item(),0)
 def test_node_training_backward(self):
  d=synthetic_node_data(0,90); model=SelectedNodeGNN(d.x.size(1))
  loss,diag=anglefair_training_step(model,d,.01)
  self.assertTrue(torch.isfinite(loss));loss.backward()
  self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
 def test_missing_sensitive_rejected(self):
  d=synthetic_node_data(0,90);d.sensitive[d.train_idx]=-1
  with self.assertRaises(ValueError):anglefair_training_step(SelectedNodeGNN(d.x.size(1)),d,.01)
if __name__=='__main__':unittest.main()
