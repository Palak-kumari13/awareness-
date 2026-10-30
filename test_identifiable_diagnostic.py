import unittest
import torch
from identifiable_motif_experiment import make_data, InterpretableSelectedGNN
from vgr.core import edge_adjacency

class TestIdentifiableDiagnostic(unittest.TestCase):
    def test_motif_is_randomized_and_label_is_structural(self):
        data=make_data(40,123)
        locations=set()
        for g in data:
            marks=g.x[:,0]
            loc=tuple(torch.where(marks==1)[0].tolist());locations.add(loc)
            self.assertEqual(len(loc),3)
            A=edge_adjacency(g)
            edges=sum(int(A[loc[i],loc[j]]>0) for i,j in [(0,1),(1,2),(0,2)])
            self.assertEqual(edges,3 if g.y==1 else 2)
            self.assertEqual(int(g.motif.sum()),edges)
        self.assertGreater(len(locations),20)
    def test_training_gradients(self):
        g=make_data(2,6)[0]
        model=InterpretableSelectedGNN()
        model(g).backward()
        self.assertTrue(model.encoder.weight.grad is not None)
        self.assertTrue(model.classifier[-1].weight.grad is not None)
    def test_edge_order_numerical_invariance(self):
        g=make_data(2,8)[0]
        model=InterpretableSelectedGNN().eval()
        with torch.no_grad():
            a=model(g)
            perm=torch.arange(len(g.edges)-1,-1,-1)
            from vgr.core import Graph
            b=model(Graph(g.x,g.edges[perm],g.y,g.sensitive,g.motif[perm]))
        self.assertLess(abs(float(a-b)),1e-3)
