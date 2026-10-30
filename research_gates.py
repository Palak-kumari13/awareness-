"""Validation-only experimental gating, adapted from the published Angle-FairGNN
Global Fallback Controller design. NOT the original implementation.
No test labels in selection. Explanation certificates are never inferred from score spread alone.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class GatePolicy:
    max_auc_drop: float = .02
    max_ba_drop: float = .02
    max_dp_gap: float = .15
    max_eo_gap: float = .15
    max_prediction_drift: float = .08
    min_auc: float = .5
    min_balanced_accuracy: float = .5
    min_score_std: float = 1e-4
    min_explanation_advantage: float = .01


def _valid(x):
    try: return x is not None and math.isfinite(float(x))
    except (TypeError,ValueError): return False


def evaluate_candidate(row,baseline,policy=GatePolicy(), require_fairness=True,require_robustness=True):
    """Rows must contain ONLY validation metrics. Missing requested metrics fail closed."""
    issues=[]
    for k in ('auroc','balanced_accuracy'):
        if not _valid(row.get(k)): issues.append('missing_'+k)
    if not issues:
        if row['auroc'] < policy.min_auc: issues.append('low_auroc')
        if row['balanced_accuracy'] < policy.min_balanced_accuracy: issues.append('low_balanced_accuracy')
        if row['auroc']+policy.max_auc_drop < baseline['auroc']: issues.append('auroc_regression')
        if row['balanced_accuracy']+policy.max_ba_drop < baseline['balanced_accuracy']: issues.append('ba_regression')
    if bool(row.get('collapsed',True)):issues.append('constant_class_prediction')
    if require_fairness:
        for k,limit in [('dp_gap',policy.max_dp_gap),('eo_gap',policy.max_eo_gap)]:
            if not _valid(row.get(k)) or row[k]>limit:issues.append('fairness_'+k)
    if require_robustness:
        k='prediction_drift'
        if not _valid(row.get(k)) or row[k]>policy.max_prediction_drift:issues.append('robustness_drift')
    return issues


def explanation_evidence(row,policy=GatePolicy()):
    """Explicit abstention when random-control or score-spread tests are missing."""
    required=('score_std','selected_fidelity','random_fidelity','degree_fidelity')
    if any(not _valid(row.get(k)) for k in required):return {'passed':False,'reason':'missing_evidence'}
    if row['score_std']<policy.min_score_std:return {'passed':False,'reason':'score_collapse'}
    improvement=min(row['random_fidelity'],row['degree_fidelity'])-row['selected_fidelity']
    if improvement <= policy.min_explanation_advantage:return {'passed':False,'reason':'no_control_superiority'}
    return {'passed':True,'reason':'validation_control_superiority_only'}


def select_checkpoint(candidates,policy=GatePolicy(),require_fairness=True,require_robustness=True):
    """Select an eligible candidate from validation records, or fall back to baseline.
    Index 0 is always baseline. Never asserts test-set nonregression.
    """
    if not candidates:raise ValueError('No candidates')
    reference=candidates[0]
    if not all(_valid(reference.get(k)) for k in ('auroc','balanced_accuracy')):
        raise ValueError('Baseline validation utility unavailable')
    review=[];eligible=[]
    for j,c in enumerate(candidates):
        issues=evaluate_candidate(c,reference,policy,require_fairness,require_robustness)
        review.append({'index':j,'rejections':issues,'explanation':explanation_evidence(c,policy)})
        if not issues:eligible.append(j)
    chosen=max(eligible,key=lambda j:(candidates[j]['auroc'],candidates[j]['balanced_accuracy'],-j)) if eligible else 0
    return {'chosen_index':chosen,'baseline_fallback':chosen==0,'candidate_reviews':review,
            'baseline_eligible':len(review[0]['rejections'])==0,
            'selection_basis':'validation_only','test_claims':False}
