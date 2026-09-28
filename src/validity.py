"""Label-free validity: internal indices and stability under resampling and under seeds.

Protocol (as in the original run):
* reported labelling: ``f(Z, base_seed)`` on all 3,607 rows;
* resampling stability: refit on 20 subsamples of 80% (drawn once with ``default_rng(12345)`` and shared by all
  candidates), mean pairwise ARI over the 190 pairs, each pair compared on the rows the two subsamples share;
* seed stability: refit with seeds 1..5, mean pairwise ARI over the 15 pairs of the 6 labellings.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .cluster import ClusterFn


@dataclass
class Refits:
    """The reported labelling plus all refits used for stability."""

    labels: np.ndarray          # (n,)
    sub_labels: np.ndarray      # (n_subsamples, m) labels of each subsample refit
    seed_labels: np.ndarray     # (n_seeds, n)


def make_subsamples(n: int, n_subsamples: int, fraction: float, rng_seed: int) -> np.ndarray:
    """Sorted row indices of each subsample, shape (n_subsamples, int(fraction * n))."""
    rng = np.random.default_rng(rng_seed)
    size = int(fraction * n)
    return np.stack([np.sort(rng.choice(n, size, replace=False)) for _ in range(n_subsamples)])


def mean_pair_ari(labels: list[np.ndarray], idxs: list[np.ndarray] | None = None) -> tuple[float, float]:
    """Mean and SD of the ARI over all pairs. With ``idxs``, each pair is compared on its shared rows."""
    scores = []
    for a, b in combinations(range(len(labels)), 2):
        if idxs is None:
            scores.append(adjusted_rand_score(labels[a], labels[b]))
            continue
        _, pa, pb = np.intersect1d(idxs[a], idxs[b], return_indices=True)
        scores.append(adjusted_rand_score(labels[a][pa], labels[b][pb]))
    return float(np.mean(scores)), float(np.std(scores))


def internal_indices(Z: np.ndarray, labels: np.ndarray) -> dict:
    """Cosine silhouette, Davies-Bouldin and Calinski-Harabasz. Noise (-1) is excluded."""
    keep = labels >= 0
    lab = labels[keep]
    if len(np.unique(lab)) < 2:
        return {"sil": np.nan, "db": np.nan, "ch": np.nan}
    return {"sil": float(silhouette_score(Z[keep], lab, metric="cosine")),
            "db": float(davies_bouldin_score(Z[keep], lab)),
            "ch": float(calinski_harabasz_score(Z[keep], lab))}


def fingerprint(spec: dict, cfg: dict, subsamples: np.ndarray, data_id: str) -> str:
    """Hash of everything a cached refit depends on."""
    payload = json.dumps({"spec": spec, "graph": cfg["graph"], "leiden": cfg["leiden"], "kmeans": cfg["kmeans"],
                          "umap_hdbscan": cfg["umap_hdbscan"], "stability": cfg["stability"],
                          "features": cfg["features"], "pca_random_state": cfg["pca_random_state"],
                          "data": data_id}, sort_keys=True)
    return hashlib.sha256(payload.encode() + subsamples.tobytes()).hexdigest()


def fit_refits(fn: ClusterFn, Z: np.ndarray, subsamples: np.ndarray, cfg: dict,
               cache: Path | None = None, cache_key: str | None = None) -> Refits:
    """Reported labelling, subsample refits and seed refits, optionally cached as .npz (no pickling).

    The cache key covers the parameters and the input data but not the code: delete cache/ after editing src/.
    """
    if cache is not None and cache.exists():
        with np.load(cache) as c:
            if str(c["fingerprint"]) == cache_key:
                return Refits(c["labels"], c["sub_labels"], c["seed_labels"])
    st = cfg["stability"]
    labels = fn(Z, st["base_seed"])
    sub_labels = np.stack([fn(Z[ix], st["base_seed"]) for ix in subsamples])
    seed_labels = np.stack([fn(Z, s) for s in st["extra_seeds"]])
    refits = Refits(labels, sub_labels, seed_labels)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, labels=labels, sub_labels=sub_labels, seed_labels=seed_labels,
                            fingerprint=np.array(cache_key))
    return refits


def validity_row(Z: np.ndarray, refits: Refits, subsamples: np.ndarray) -> dict:
    """One row of validity.csv: size profile, internal indices, resampling and seed stability."""
    lab = refits.labels
    sizes = np.bincount(lab[lab >= 0])
    sub_mean, sub_sd = mean_pair_ari(list(refits.sub_labels), list(subsamples))
    seed_mean, seed_sd = mean_pair_ari([lab] + list(refits.seed_labels))
    return {"n_clusters": int(len(sizes[sizes > 0])), "noise_frac": float((lab < 0).mean()),
            "largest_frac": float(sizes.max() / len(lab)), **internal_indices(Z, lab),
            "stab_sub_ari": sub_mean, "stab_sub_sd": sub_sd, "n_sub": len(subsamples),
            "stab_seed_ari": seed_mean, "stab_seed_sd": seed_sd, "n_seed_labellings": 1 + len(refits.seed_labels)}
