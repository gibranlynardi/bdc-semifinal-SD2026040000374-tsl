"""Clustering methods. Each method is a function ``f(Z, seed) -> labels`` (noise, if any, is -1)."""
from __future__ import annotations

import warnings
from functools import partial
from typing import Callable

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans

ClusterFn = Callable[[np.ndarray, int], np.ndarray]


def knn_graph(Z: np.ndarray, n_neighbors: int, metric: str, random_state: int):
    """scanpy kNN graph (UMAP connectivities) on the rows of ``Z``, returned inside an AnnData.

    For fewer than 4,096 rows scanpy computes exact neighbours, so the graph is deterministic.
    """
    import anndata as ad
    import scanpy as sc

    adata = ad.AnnData(np.asarray(Z, dtype=np.float32))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X", metric=metric, random_state=random_state)
    return adata


def leiden(Z: np.ndarray, seed: int, *, resolution: float, graph: dict, leiden_cfg: dict) -> np.ndarray:
    """Leiden community detection on the kNN graph. The graph seed stays fixed, ``seed`` drives Leiden."""
    import scanpy as sc

    adata = knn_graph(Z, graph["n_neighbors"], graph["metric"], graph["random_state"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sc.tl.leiden(adata, resolution=resolution, flavor=leiden_cfg["flavor"],
                     n_iterations=leiden_cfg["n_iterations"], directed=leiden_cfg["directed"],
                     random_state=seed)
    return adata.obs["leiden"].astype(int).to_numpy()


def kmeans(Z: np.ndarray, seed: int, *, k: int, n_init: int) -> np.ndarray:
    """K-Means (Euclidean on the L2-normalised rows, i.e. spherical up to scale)."""
    return KMeans(k, n_init=n_init, random_state=seed).fit_predict(Z)


def umap_hdbscan(Z: np.ndarray, seed: int, *, params: dict) -> np.ndarray:
    """UMAP to a low dimension, then HDBSCAN. Noise points keep label -1."""
    import umap

    with warnings.catch_warnings():  # a fixed random_state makes UMAP single-threaded, which it announces
        warnings.filterwarnings("ignore", message="n_jobs value", category=UserWarning)
        emb = umap.UMAP(n_neighbors=params["n_neighbors"], n_components=params["n_components"],
                        min_dist=params["min_dist"], metric=params["metric"], random_state=seed).fit_transform(Z)
    return HDBSCAN(min_cluster_size=params["min_cluster_size"],  # copy=False is the scikit-learn 1.8 default
                   cluster_selection_method=params["cluster_selection_method"], copy=False).fit_predict(emb)


def make_method(spec: dict, cfg: dict) -> ClusterFn:
    """Turn a candidate entry of config.yaml into a clustering function ``f(Z, seed)``."""
    method = spec["method"]
    if method == "leiden":
        return partial(leiden, resolution=spec["resolution"], graph=cfg["graph"], leiden_cfg=cfg["leiden"])
    if method == "kmeans":
        return partial(kmeans, k=spec["k"], n_init=cfg["kmeans"]["n_init"])
    if method == "umap_hdbscan":
        return partial(umap_hdbscan, params=cfg["umap_hdbscan"])
    raise ValueError(f"unknown method {method!r}")


def method_label(spec: dict) -> str:
    """Human-readable method name, as in the original internal.csv."""
    if spec["method"] == "leiden":
        return f"Leiden r={spec['resolution']}"
    if spec["method"] == "kmeans":
        return f"KMeans k={spec['k']}"
    return "UMAP+HDBSCAN"
