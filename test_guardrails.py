import unittest
import numpy as np
from vgr.guardrails import select_validation_threshold,evaluate
class TestGuardrails(unittest.TestCase):
    def test_threshold_not_constant_when_separable(self):
        y=np.array([0,0,1,1]);p=np.array([.4,.45,.47,.48]);t=select_validation_threshold(y,p)
        self.assertAlmostEqual(evaluate(y,np.array([0,1,0,1]),p,t)['balanced_accuracy'],1.)
    def test_constant_detection(self):
        r=evaluate(np.array([0,1,0,1]),np.array([0,0,1,1]),np.array([.8,.8,.8,.8]))
        self.assertTrue(r['collapsed'])
    def test_threshold_is_deterministic(self):
        y=np.array([0,1,0,1]);p=np.array([.1,.9,.2,.8]);self.assertEqual(select_validation_threshold(y,p),select_validation_threshold(y,p))
if __name__=='__main__':unittest.main()
