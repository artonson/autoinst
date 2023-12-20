import numpy as np
from scipy import sparse
import torch.nn.functional as F


def normalize_mat(A, eps=1e-5):
    A -= np.min(A[np.nonzero(A)]) if np.any(A > 0) else 0
    A[A < 0] = 0.
    A /= A.max() + eps
    return A

def cut_cost(W, mask):
    return (np.sum(W) - np.sum(W[mask][:, mask]) - np.sum(W[~mask][:, ~mask])) / 2

def ncut_cost(W, D, cut):
    cost = cut_cost(W, cut)
    assoc_a = D.todense()[cut].sum() # Anastasiia: this also can be optimized in the future
    assoc_b = D.todense()[~cut].sum()
    return (cost / assoc_a) + (cost / assoc_b)

def get_affinity_matrix(feats, tau=0.15, eps=1e-5, normalize_sim=True, similarity_metric='cos'):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats_a, feats_b = feats
    feats_a = F.normalize(feats_a, p=2, dim=-1)
    feats_b = F.normalize(feats_b, p=2, dim=-1)
    A_a, A_b = feats_a @ feats_a.T, feats_b @ feats_b.T

    # Normalize both attentions if requested
    A_a, A_b = A_a.cpu().numpy(), A_b.cpu().numpy()
    A_a = normalize_mat(A_a) if normalize_sim else A_a
    A_b = normalize_mat(A_b) if normalize_sim else A_b

    # Combine attentions
    A = (A_a + A_b) / 2

    # convert the affinity matrix to a binary one.
    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=0)
    D = np.diag(d_i)
    return A, D

def get_min_ncut(ev, d, w, num_cuts):
    mcut = np.inf
    mn = ev.min()
    mx = ev.max()

    # If all values in `ev` are equal, it implies that the graph can't be
    # further sub-divided. In this case the bi-partition is the the graph
    # itself and an empty set.
    min_mask = np.zeros_like(ev, dtype=bool)
    if np.allclose(mn, mx):
        return min_mask, mcut

    # Refer Shi & Malik 2001, Section 3.1.3, Page 892
    # Perform evenly spaced n-cuts and determine the optimal one.
    for t in np.linspace(mn, mx, num_cuts, endpoint=False):
        mask = ev > t
        cost = ncut_cost(w,d, mask)
        if cost < mcut:
            min_mask = mask
            mcut = cost

    return min_mask, mcut


def normalized_cut(w, labels, T = 0.01):
    W = w + sparse.identity(w.shape[0])

    if W.shape[0] > 2:
        d = np.array(W.sum(axis=0))[0]
        d2 = np.reciprocal(np.sqrt(d))
        D = sparse.diags(d)
        D2 = sparse.diags(d2)

        A = D2 * (D - W) * D2

        eigvals, eigvecs = sparse.linalg.eigsh(A, 2, sigma = 1e-10, which='LM')

        index2 = np.argsort(eigvals)[1]

        ev = eigvecs[:, index2]
        mask, mcut = get_min_ncut(ev, D, w, 10)

        if mcut < T:
            labels1 = normalized_cut(w[mask][:, mask], labels[mask], T=T)
            labels2 = normalized_cut(w[~mask][:, ~mask], labels[~mask], T=T)
            return labels1 + labels2
        else:
            return [labels]
    else:
        return [labels]
