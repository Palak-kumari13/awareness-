"""Task-agnostic structural awareness for undirected, simple graphs.
All diagnostics use X and adjacency only; no labels, group memberships or planted-motif indicators.
"""
import numpy as np
import torch

def structural_features(edges, n, mode='topology', spectral_rank=3):
    if mode not in ('topology','motif','spectral','hybrid'):
        raise ValueError(mode)
    a = torch.zeros((n,n),dtype=torch.float32,device=edges.device)
    if edges.numel():
        u,v=edges[:,0],edges[:,1]
        a[u,v]=1.;a[v,u]=1.
    deg=a.sum(1)
    cn=a@a
    triangles=torch.diag(cn@a)/2
    wedge=(deg*(deg-1)/2).clamp_min(1)
    clustering=triangles/wedge
    dnorm=deg/max(n-1,1)
    local=torch.stack([dnorm,clustering,triangles/max(n*n,1)],dim=-1)
    if mode in ('spectral','hybrid'):
        # L = I - D^-1/2 A D^-1/2; eigenvectors enter only through
        # sign-invariant pairwise squared distances. Eigenspace instability at
        # repeated eigenvalues remains a limitation and is explicitly documented.
        dinv=deg.clamp_min(1).rsqrt()
        lap=torch.eye(n,device=a.device)-dinv[:,None]*a*dinv[None,:]
        _,v=torch.linalg.eigh(lap)
        k=min(spectral_rank,max(n-1,0))
        pe=v[:,1:1+k] if k else v[:,:0]
        if pe.shape[1]<spectral_rank:
            pe=torch.nn.functional.pad(pe,(0,spectral_rank-pe.shape[1]))
        local=torch.cat([local,pe.square()],-1) if mode=='spectral' else local
    if not len(edges):
        width={'topology':6,'motif':7,'spectral':7,'hybrid':11}[mode]
        return local,torch.zeros((0,width),device=a.device),a
    u,v=edges[:,0],edges[:,1]
    base=torch.stack([dnorm[u]+dnorm[v],(dnorm[u]-dnorm[v]).abs(),cn[u,v]/max(n-2,1),clustering[u]+clustering[v],(clustering[u]-clustering[v]).abs(),(deg[u]+deg[v])/(2*max(n-1,1))],-1)
    if mode=='topology': out=base
    elif mode=='motif': out=torch.cat([base,(triangles[u]+triangles[v]).unsqueeze(-1)/max(n*n,1)],-1)
    elif mode=='spectral': out=torch.cat([base,(pe[u]-pe[v]).square().sum(-1,keepdim=True)],-1)
    else: out=torch.cat([base,(triangles[u]+triangles[v]).unsqueeze(-1)/max(n*n,1),(pe[u]-pe[v]).square(),(pe[u]-pe[v]).square().sum(-1,keepdim=True)],-1)
    return local,out,a
