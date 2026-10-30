import unittest
import numpy as np
from vgr.decision import classification_audit,choose_threshold

class DecisionAuditTests(unittest.TestCase):
    def test_all_positive_is_failure_even_when_gap_zero(self):
        result=classification_audit([0,0,1,1],[.7,.8,.9,.95])
        self.assertTrue(result['collapsed_predictions'])
        self.assertFalse(result['gate_pass'])
        self.assertEqual(result['balanced_accuracy'],.5)
    def test_validation_threshold_can_recover_classes(self):
        selected=choose_threshold([0,0,1,1],[.55,.6,.7,.8])
        self.assertTrue(selected['class_recall_constraint_feasible'])
        r=classification_audit([0,0,1,1],[.55,.6,.7,.8],selected['threshold'])
        self.assertTrue(r['gate_pass'])
    def test_constant_probabilities_cannot_fake_discrimination(self):
        selected=choose_threshold([0,0,1,1],[.7,.7,.7,.7])
        self.assertFalse(selected['class_recall_constraint_feasible'])
        r=classification_audit([0,0,1,1],[.7]*4,selected['threshold'])
        self.assertFalse(r['gate_pass'])
    def test_single_class_validation_rejected(self):
        with self.assertRaises(ValueError):choose_threshold([1,1,1],[.6,.7,.8])
    def test_test_one_class_not_claimed_valid(self):
        r=classification_audit([1,1],[.8,.9])
        self.assertFalse(r['gate_pass'])
    def test_confusion_counts(self):
        r=classification_audit([0,0,1,1],[.1,.8,.9,.2])
        self.assertEqual(r['confusion_matrix'],[[1,1],[1,1]])

if __name__=='__main__':unittest.main()
