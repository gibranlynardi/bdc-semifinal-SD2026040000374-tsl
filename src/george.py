"""Subclass estimation per coarse name, following GEORGE Step 1 (Sohoni et al., NeurIPS 2020, Sec. 4.1 and App. B.3.4).

This is a robustness check on the main Leiden result, not part of the label-free choice. It partitions the photos
by their coarse name (the filename name for catalogue photos, or the confident nearest-neighbour name vote for all
photos) and clusters each partition separately, as GEORGE does for each superclass.

Two variants, both from the paper:
* ``bit``  : raw L2 features + k-means (n_init=3), as in George-BiT (Sec. 6.5, App. B.3.6), our frozen-feature analogue.
* ``umap`` : UMAP to 2-D (n_neighbors=10, min_dist=0) + GMM (n_init=3, full covariance), the paper default.
k in 2..10 is chosen by the highest average per-cluster silhouette (ties go to the larger k), then each cluster is
over-clustered into 5 and a sub-cluster is kept if its mean silhouette rises above max(old, 0) and it has at least
max(0.5% of the partition, 25) photos. The code's extra ERM-loss filter needs a trained classifier and is not applied.
GEORGE Step 2 (GDRO training) is not used: nothing is trained here.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture


def _fit(X, k, algo, seed):
    if algo == "bit":
        return KMeans(k, n_init=3, random_state=seed).fit_predict(X)
    return GaussianMixture(k, n_init=3, random_state=seed).fit_predict(X)


def _avg_cluster_sil(X, lab):
    s = silhouette_samples(X, lab)
    return float(np.mean([s[lab == c].mean() for c in np.unique(lab)]))


def reduce(Z: np.ndarray, algo: str, seed: int) -> np.ndarray:
    if algo == "bit":
        return Z
    import umap
    return umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.0, random_state=seed).fit_transform(Z)


def estimate_subclasses(X: np.ndarray, algo: str, seed: int = 0, k_max: int = 10, factor: int = 5):
    """Return (k chosen by silhouette, its average per-cluster silhouette, final labels after over-clustering)."""
    best = None
    for k in range(2, k_max + 1):
        if k >= len(X):
            break
        lab = _fit(X, k, algo, seed)
        sc = _avg_cluster_sil(X, lab)
        if best is None or sc >= best[1]:
            best = (k, sc, lab)
    k, sc, lab = best
    lab = lab.copy()
    n_min = max(int(0.005 * len(X)), 25)
    old = silhouette_samples(X, lab)
    nxt = lab.max() + 1
    for c in range(k):
        m = np.where(lab == c)[0]
        if len(m) < 2 * factor:
            continue
        sub = _fit(X[m], factor, algo, seed)
        trial = lab.copy()
        trial[m] = nxt + sub
        new = silhouette_samples(X, trial)
        for j in range(factor):
            mm = m[sub == j]
            if len(mm) >= n_min and new[mm].mean() > max(old[mm].mean(), 0):
                lab[mm] = nxt + j
        nxt += factor
    _, lab = np.unique(lab, return_inverse=True)
    return k, sc, lab
