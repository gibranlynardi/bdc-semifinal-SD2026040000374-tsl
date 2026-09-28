"""Post-hoc audit of the chosen clustering against the filename names.

This module runs AFTER the label-free choice. The filename name (``device_label``: 10 coarse device types)
exists only for the 2,332 catalogue photos. It is used here to describe the clusters, never to form or pick them.
Human annotation labels are not read anywhere in the package.

* name vote: each scraped photo gets the majority filename name of its 15 nearest catalogue photos
  (cosine on uncentred L2 embeddings), with confidence = votes / 15. It is a description aid only.
* split score of a name: normalised entropy of how its photos spread over the clusters,
  H / ln(K) with K the number of clusters (0 = one cluster, 1 = uniform over all clusters).
* orphan: a cluster whose confident majority name covers less than 50% of its members
  (members without a confident name count in the denominator).
* refit stability: in each of the 25 refits (20 subsamples of 80% + 5 seeds), the share of a reference
  cluster that lands in its top refit cluster, and whether paired clusters land in different refit clusters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .validity import Refits


def name_vote(Z_uncentred: np.ndarray, names: np.ndarray, catalogue: np.ndarray, scraped: np.ndarray,
              k: int) -> tuple[np.ndarray, np.ndarray]:
    """kNN majority vote of catalogue filename names for scraped photos.

    Returns ``pred`` ("" for catalogue rows) and ``conf`` (NaN for catalogue rows). Ties go to the name that
    sorts first, as ``np.unique`` orders names.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Z_uncentred[catalogue])
    _, nb = nn.kneighbors(Z_uncentred[scraped])
    cat_names = names[catalogue]
    pred = np.full(len(names), "", dtype=object)
    conf = np.full(len(names), np.nan)
    votes = []
    for row in nb:
        uniq, counts = np.unique(cat_names[row], return_counts=True)
        votes.append((uniq[counts.argmax()], counts.max() / k))
    pred[scraped] = [v[0] for v in votes]
    conf[scraped] = [v[1] for v in votes]
    return pred, conf


def confident_names(filename_names: np.ndarray, pred: np.ndarray, conf: np.ndarray, catalogue: np.ndarray,
                    threshold: float) -> np.ndarray:
    """Filename name for catalogue photos, voted name for scraped photos with conf >= threshold, else ""."""
    voted = np.where(np.nan_to_num(conf, nan=0.0) >= threshold, pred, "")
    return np.where(catalogue, filename_names, voted).astype(object)


def crosstab(labels: np.ndarray, names: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Cluster x name counts on the rows in ``mask`` (clusters as rows, all clusters kept)."""
    ct = pd.crosstab(pd.Series(labels[mask], name="cluster"), pd.Series(names[mask], name="name"))
    return ct.reindex(sorted(np.unique(labels)), fill_value=0)


def _majority(names: pd.Series) -> pd.Series:
    """Counts of non-empty names, sorted by count (descending) then name, for deterministic ties."""
    vc = names[names != ""].value_counts()
    return vc.sort_index().sort_values(ascending=False, kind="stable")


def split_scores(labels: np.ndarray, names: np.ndarray, mask: np.ndarray, min_share: float, view: str,
                 majority_of: dict[str, list[int]] | None = None) -> pd.DataFrame:
    """One row per name: normalised entropy over clusters and the clusters that hold >= ``min_share`` of it."""
    ct = crosstab(labels, names, mask & (names != ""))
    n_clusters = len(np.unique(labels[labels >= 0]))
    rows = []
    for name in ct.columns:
        counts = ct[name].to_numpy()
        p = counts / counts.sum()
        nz = p[p > 0]
        entropy = float(-(nz * np.log(nz)).sum())
        order = np.argsort(-p, kind="stable")
        parts = [(int(ct.index[j]), p[j]) for j in order if p[j] >= min_share]
        rows.append({"view": view, "name": name, "n": int(counts.sum()),
                     "split_score": entropy / np.log(n_clusters), "effective_clusters": float(np.exp(entropy)),
                     "top_cluster": int(ct.index[order[0]]), "top_share": float(p[order[0]]),
                     "n_parts": len(parts), "parts": " ".join(f"{c}:{s:.3f}" for c, s in parts),
                     "split": len(parts) >= 2,
                     "majority_in_clusters": " ".join(map(str, (majority_of or {}).get(name, [])))})
    return pd.DataFrame(rows)


def orphan_table(labels: np.ndarray, conf_names: np.ndarray, catalogue: np.ndarray,
                 max_share: float) -> pd.DataFrame:
    """One row per cluster with its confident majority name. ``orphan`` = majority share < ``max_share``."""
    rows = []
    for c in sorted(np.unique(labels)):
        m = labels == c
        vc = _majority(pd.Series(conf_names[m]))
        n_named = int(vc.sum())
        top, top_n = (vc.index[0], int(vc.iloc[0])) if n_named else ("", 0)
        second, second_n = (vc.index[1], int(vc.iloc[1])) if len(vc) > 1 else ("", 0)
        rows.append({"cluster": int(c), "size": int(m.sum()), "n_catalogue": int((m & catalogue).sum()),
                     "catalogue_share": float((m & catalogue).sum() / m.sum()),
                     "n_named": n_named, "majority_name": top, "majority_n": top_n,
                     "majority_share": top_n / m.sum(),
                     "majority_share_of_named": top_n / n_named if n_named else np.nan,
                     "second_name": second, "second_share": second_n / m.sum(),
                     "orphan": top_n / m.sum() < max_share})
    return pd.DataFrame(rows)


def majority_map(orphans: pd.DataFrame) -> dict[str, list[int]]:
    """name -> clusters whose confident majority name it is."""
    out: dict[str, list[int]] = {}
    for c, name in zip(orphans.cluster, orphans.majority_name):
        out.setdefault(name, []).append(int(c))
    return out


def refit_stability(ref: np.ndarray, refits: Refits, subsamples: np.ndarray, groups: dict[str, int],
                    pairs: list[list[str]], hold_share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cluster and per-pair stability over the subsample and seed refits of the chosen configuration."""
    n = len(ref)
    runs = [(ix, lab) for ix, lab in zip(subsamples, refits.sub_labels)]
    runs += [(np.arange(n), lab) for lab in refits.seed_labels]
    clusters = sorted(np.unique(ref))
    share = {c: [] for c in clusters}
    top = {c: [] for c in clusters}
    for ix, lab in runs:
        r = ref[ix]
        for c in clusters:
            vc = pd.Series(lab[r == c]).value_counts(normalize=True)
            share[c].append(vc.iloc[0])
            top[c].append(vc.index[0])
    name_of = {c: g for g, c in groups.items()}
    per_cluster = pd.DataFrame([{
        "cluster": int(c), "group": name_of.get(c, ""), "n": int((ref == c).sum()),
        "mean_top_share": float(np.mean(share[c])), "min_top_share": float(np.min(share[c])),
        "frac_refits_hold": float(np.mean(np.array(share[c]) >= hold_share)), "n_refits": len(runs),
    } for c in clusters])
    per_pair = pd.DataFrame([{
        "pair": f"{a} vs {b}", "cluster_a": groups[a], "cluster_b": groups[b],
        "frac_refits_apart": float(np.mean([x != y for x, y in zip(top[groups[a]], top[groups[b]])])),
        "n_refits": len(runs),
    } for a, b in pairs])
    return per_cluster, per_pair
