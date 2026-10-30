"""Experimental mask-independent group-conditional principal-angle loss.

For each target label, compare the representation subspaces of sensitive groups.
This is a geometric group-alignment surrogate, NOT an established fairness guarantee.
Skipped groups/classes with fewer than min_group examples cannot supply subspace evidence.
"""
import torch
import torch.nn.functional as F


def principal_angle_alignment(embeddings, labels, sensitive, min_group=3, rank=2, eps=1e-8):
    if embeddings.ndim != 2 or labels.ndim != 1 or sensitive.ndim != 1:
        raise ValueError('Expected [N,D] embeddings and [N] labels/groups')
    if len(embeddings) != len(labels) or len(labels) != len(sensitive):
        raise ValueError('Inconsistent lengths')
    if rank < 1 or min_group < 2:
        raise ValueError('rank>=1 and min_group>=2 required')
    if not torch.isfinite(embeddings).all():
        raise ValueError('Nonfinite embeddings')
    terms=[]
    for label in (0,1):
        sub=[]
        for group in (0,1):
            h=embeddings[(labels==label)&(sensitive==group)]
            if len(h)<min_group:break
            centered=h-h.mean(0,keepdim=True)
            # Small jitter-free SVD; rank fixed by group support.
            _,sv,vh=torch.linalg.svd(centered,full_matrices=False)
            r=min(rank,len(h)-1,h.shape[1]);
            if r<1 or sv[r-1].detach()<=eps:break
            sub.append(vh[:r].T)
        if len(sub)==2:
            r=min(sub[0].shape[1],sub[1].shape[1])
            cosine=torch.linalg.svdvals(sub[0][:,:r].T@sub[1][:,:r]).clamp(0.,1.)
            terms.append(1-cosine.square().mean())
    if not terms:
        return embeddings.sum()*0.,0
    return torch.stack(terms).mean(),len(terms)


def anglefair_training_step(model,data,weight=0.01,base_loss=None):
    """Add geometric alignment to existing node classification; zero is a no-op."""
    if base_loss is None:
        logits,embed=model(data,return_embeddings=True)
        base_loss=F.binary_cross_entropy_with_logits(logits[data.train_idx],data.y[data.train_idx].float())
    else:
        _,embed=model(data,return_embeddings=True)
    if weight==0:return base_loss, {'angle_active_classes':0, 'angle_loss':0.0}
    idx=data.train_idx
    if (data.sensitive[idx]<0).any():raise ValueError('AngleFair requires known TRAIN sensitive groups')
    extra,active=principal_angle_alignment(embed[idx],data.y[idx],data.sensitive[idx])
    return base_loss+weight*extra, {'angle_active_classes':active,'angle_loss':float(extra.detach())}
