import unittest,torch
from vgr.selector_repair import randomized_motifs,frozen_split,soft_binary_kl,train_selector_variant,explain_precision
from vgr.core import GNN
class SelectorRepairTests(unittest.TestCase):
    def test_randomized_positions(self):
        a=randomized_motifs(30,seed=1)
        assert len({tuple(sorted(set(a0.edges[a0.motif].flatten().tolist()))) for a0 in a})>1
        assert set(g.y for g in a)=={0,1}
    def test_deterministic(self):
        a=randomized_motifs(8,seed=7);b=randomized_motifs(8,seed=7)
        assert all(torch.equal(x.edges,y.edges) and torch.equal(x.motif,y.motif) for x,y in zip(a,b))
    def test_disjoint_generation(self):
        a,b,c=frozen_split(0,20)
        assert a[0].edges.tolist()!=b[0].edges.tolist()
    def test_binary_kl(self):
        x=torch.tensor([-.3,.8],requires_grad=True)
        assert abs(float(soft_binary_kl(x,x)))<1e-6
        soft_binary_kl(x,torch.zeros_like(x,requires_grad=True)).backward()
    def test_methods(self):
        tr,va,te=frozen_split(0,12)
        for mode in ['joint','alternate','teacher']:
            model=GNN(1,hidden=8)
            ref=GNN(1,hidden=8,selector=False)
            model,history=train_selector_variant(model,tr,va,method=mode,reference=ref if mode=='teacher' else None,epochs=2)
            self.assertTrue(len(history)>0)
            self.assertEqual(explain_precision(model,te,2)['n_graphs'],len(te))
if __name__=='__main__':unittest.main()
