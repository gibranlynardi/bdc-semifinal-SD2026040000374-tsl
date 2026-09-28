"""The label-free choice of one configuration (written down before any evaluation).

Rule: among configurations with at least ``min_clusters`` non-noise clusters and a resampling-stability
estimate, choose the highest resampling stability (mean pairwise ARI of refits on 80% subsamples, compared
on shared rows). If another configuration is within ``tie_margin`` of the best, prefer the higher cosine
silhouette. Only the label-free columns of validity.csv are read.
"""
from __future__ import annotations

import pandas as pd

RULE = ("Among all configurations with >= {min_clusters} non-noise clusters and a resampling-stability estimate, "
        "choose the one with the highest resampling stability (mean pairwise ARI over 80% subsamples, compared on "
        "shared points). If two are within {tie_margin} of each other, prefer the higher cosine silhouette. "
        "Only label-free quantities are read.")


def choose(table: pd.DataFrame, min_clusters: int, tie_margin: float) -> tuple[str, pd.DataFrame]:
    """Return the chosen key and the table with ``eligible``, ``rank_stability`` and ``chosen`` columns."""
    t = table.copy()
    t["eligible"] = (t.n_clusters >= min_clusters) & t.stab_sub_ari.notna()
    cand = t[t.eligible].sort_values("stab_sub_ari", ascending=False, kind="stable")
    if cand.empty:
        raise ValueError("no eligible configuration")
    best = cand.stab_sub_ari.iloc[0]
    tied = cand[cand.stab_sub_ari >= best - tie_margin]
    key = tied.sort_values("sil", ascending=False, kind="stable").key.iloc[0]
    t["rank_stability"] = t.stab_sub_ari.where(t.eligible).rank(ascending=False, method="min").astype("Int64")
    t["within_tie_margin"] = t.key.isin(tied.key)
    t["chosen"] = t.key == key
    return key, t


def choice_markdown(key: str, table: pd.DataFrame, min_clusters: int, tie_margin: float) -> str:
    """A short markdown record of the choice (no timestamp, so reruns are byte-identical)."""
    row = table.set_index("key").loc[key]
    cols = ["key", "source", "n_clusters", "stab_sub_ari", "n_sub", "stab_seed_ari", "sil"]
    cand = table[table.eligible].sort_values("stab_sub_ari", ascending=False, kind="stable")[cols].round(3)
    rest = table.loc[~table.eligible, "key"].tolist()
    return ("# Label-free choice of the configuration\n\nOnly label-free columns were read.\n\n"
            f"**Rule:** {RULE.format(min_clusters=min_clusters, tie_margin=tie_margin)}\n\n"
            f"**Chosen:** `{key}` ({row.features}, {row.method}): {int(row.n_clusters)} clusters, "
            f"resampling ARI {row.stab_sub_ari:.3f}, seed ARI {row.stab_seed_ari:.3f}, "
            f"cosine silhouette {row.sil:.3f}.\n\n"
            f"Eligible candidates, sorted by resampling stability:\n\n{cand.to_markdown(index=False)}\n\n"
            f"Not eligible (fewer than {min_clusters} clusters or no resampling estimate): "
            f"{', '.join(rest) if rest else 'none'}.\n")
