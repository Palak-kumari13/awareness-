"""FRME-inspired independent graph intervention controls.
These diagnostics do NOT claim reconstruction, causal fidelity or certified robustness.
Works with the existing undirected Graph and frozen predictor.
"""
import torch

def edge_deletion_effect(model,graph,edge_indices,probability=True):
    """Remove graph edge rows (undirected unique-edge convention), preserve node IDs/features.
    Returns baseline/ablated probability and signed original-class probability drop.
    """
    from .core import Graph
    ids=set(int(j) for j in edge_indices)
    if any(j<0 or j>=len(graph.edges) for j in ids):raise IndexError('edge index outside graph')
    model.eval()
    with torch.no_grad():
        before=torch.sigmoid(model(graph).reshape(-1)[0]).item()
        remaining=torch.tensor([j for j in range(len(graph.edges)) if j not in ids],dtype=torch.long)
        modified=Graph(graph.x,graph.edges[remaining],graph.y)
        after=torch.sigmoid(model(modified).reshape(-1)[0]).item()
    cls=int(before>=.5)
    original_prob=before if cls else 1-before
    new_prob=after if cls else 1-after
    return {'removed_edge_indices':sorted(ids),'prediction_class':cls,'before_p1':before,'after_p1':after,
            'drop_original_class_probability':original_prob-new_prob,'flipped':bool((after>=.5)!=bool(cls))}


def joint_edge_interaction(model,graph,i,j):
    """Difference-in-differences for removing two edges (model dependence, not causality)."""
    a=edge_deletion_effect(model,graph,[i])['drop_original_class_probability']
    b=edge_deletion_effect(model,graph,[j])['drop_original_class_probability']
    both=edge_deletion_effect(model,graph,[i,j])['drop_original_class_probability']
    return {'first':a,'second':b,'joint':both,'interaction':both-a-b}
