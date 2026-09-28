# Label-free choice of the configuration

Only label-free columns were read.

**Rule:** Among all configurations with >= 10 non-noise clusters and a resampling-stability estimate, choose the one with the highest resampling stability (mean pairwise ARI over 80% subsamples, compared on shared points). If two are within 0.01 of each other, prefer the higher cosine silhouette. Only label-free quantities are read.

**Chosen:** `raw__Leiden_r1.0` (raw, Leiden r=1.0): 19 clusters, resampling ARI 0.902, seed ARI 0.948, cosine silhouette 0.213.

Eligible candidates, sorted by resampling stability:

| key                  | source                                      |   n_clusters |   stab_sub_ari |   n_sub |   stab_seed_ari |    sil |
|:---------------------|:--------------------------------------------|-------------:|---------------:|--------:|----------------:|-------:|
| raw__Leiden_r1.0     | computed                                    |           19 |          0.902 |      20 |           0.948 |  0.213 |
| raw__KMeans_k10      | computed                                    |           10 |          0.899 |      20 |           0.93  |  0.2   |
| raw__Leiden_r0.5     | computed                                    |           17 |          0.89  |      20 |           0.95  |  0.212 |
| raw__KMeans_k20      | computed                                    |           20 |          0.836 |      20 |           0.823 |  0.208 |
| raw__UMAP_HDBSCAN    | computed                                    |           51 |          0.771 |      20 |           0.74  |  0.174 |
| defl20__Leiden_r1.0  | computed                                    |           24 |          0.711 |      20 |           0.891 | -0.01  |
| defl20__Leiden_r0.5  | computed                                    |           13 |          0.708 |      20 |           0.911 | -0.016 |
| raw__TEMI_K10        | recorded (original TEMI run, not re-fitted) |           10 |          0.532 |       3 |           0.481 |  0.169 |
| defl20__UMAP_HDBSCAN | computed                                    |           78 |          0.366 |      20 |           0.615 |  0.073 |
| defl20__KMeans_k10   | computed                                    |           10 |          0.196 |      20 |           0.221 |  0.019 |
| defl20__KMeans_k20   | computed                                    |           20 |          0.191 |      20 |           0.212 |  0.025 |

Not eligible (fewer than 10 clusters or no resampling estimate): raw__TEMI_K20, defl20__TEMI_K10.
