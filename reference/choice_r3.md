# Label-free choice of the cluster-card configuration

Written 2026-09-25 08:30:09 +0700, BEFORE the audit stage (no device, provenance or human label has been read by run.py yet).

**Rule (fixed before looking):** Among all configurations with >= 10 non-noise clusters and a resampling-stability estimate, choose the one with the highest resampling stability (mean pairwise ARI over 80% subsamples, compared on shared points); if two are within 0.01, prefer the higher cosine silhouette. Only internal.csv (label-free) is read.

**Chosen:** `raw__Leiden_r1.0` (raw, Leiden r=1.0); 19 clusters, resampling ARI 0.902, silhouette 0.213.

Candidates (sorted by resampling stability):

| key                  |   n_clusters |   stab_sub_ari |   n_sub |   stab_seed_ari |    sil |
|:---------------------|-------------:|---------------:|--------:|----------------:|-------:|
| raw__Leiden_r1.0     |           19 |          0.902 |      20 |           0.948 |  0.213 |
| raw__KMeans_k10      |           10 |          0.899 |      20 |           0.93  |  0.2   |
| raw__Leiden_r0.5     |           17 |          0.89  |      20 |           0.95  |  0.212 |
| raw__KMeans_k20      |           20 |          0.836 |      20 |           0.823 |  0.208 |
| raw__UMAP_HDBSCAN    |           51 |          0.771 |      20 |           0.74  |  0.174 |
| defl20__Leiden_r1.0  |           24 |          0.711 |      20 |           0.891 | -0.01  |
| defl20__Leiden_r0.5  |           13 |          0.708 |      20 |           0.911 | -0.016 |
| raw__TEMI_K10        |           10 |          0.532 |       3 |           0.481 |  0.169 |
| defl20__UMAP_HDBSCAN |           78 |          0.366 |      20 |           0.615 |  0.073 |
| defl20__KMeans_k10   |           10 |          0.196 |      20 |           0.221 |  0.019 |
| defl20__KMeans_k20   |           20 |          0.191 |      20 |           0.212 |  0.025 |

Not eligible (fewer than 10 clusters or no resampling estimate): none.

Note: TEMI resampling stability uses a reduced protocol (3 subsamples, 4 heads) because of CPU budget; only TEMI raw K=10 has one; the other TEMI subsample runs were stopped (memory thrashing / CPU budget), so those rows have none.

## Addendum (08:31, after the choice, before re-running the audit)
Rows added to internal.csv after this choice was written: `raw__TEMI_K20` and `defl20__TEMI_K10` (their TEMI fits finished later).
Neither has a resampling-stability estimate, so under the rule above neither could have been chosen; the choice stands.
`defl20__TEMI_K20` was not run (CPU/memory budget).
