"""Tidy export for the figures. Plotting itself is done by a separate script.

``clusters.csv`` has one row per analysed photo:
file, thumb, provenance, cluster, filename_name, pred_name, pred_conf, confident_name, fa2_x, fa2_y.
The ForceAtlas2 (FA2) positions are a layout of the raw kNN graph (cosine, k=15) and are for drawing only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fa2_layout(Z_float32: np.ndarray, cfg: dict) -> np.ndarray:
    """ForceAtlas2 layout of the kNN graph via scanpy (needs the ``fa2-modified`` package, about 90 s)."""
    import anndata as ad
    import scanpy as sc

    lc = cfg["layout"]
    adata = ad.AnnData(Z_float32.copy())
    sc.pp.neighbors(adata, n_neighbors=lc["n_neighbors"], use_rep="X", metric=lc["metric"],
                    random_state=lc["neighbors_random_state"])
    sc.tl.draw_graph(adata, layout="fa", random_state=lc["fa_random_state"], maxiter=lc["maxiter"])
    return np.asarray(adata.obsm["X_draw_graph_fa"], dtype=np.float64)


def load_layout(Z_float32: np.ndarray, cfg: dict, reference_dir: Path) -> tuple[np.ndarray, str]:
    """Recompute the layout or read the stored one, as ``cfg['layout']['source']`` says."""
    if cfg["layout"]["source"] == "reference":
        return np.load(reference_dir / "pos_raw.npy"), "reference/pos_raw.npy"
    return fa2_layout(Z_float32, cfg), "recomputed"


def thumb_name(file: str) -> str:
    """Thumbnails are stored as <stem>.jpg whatever the original extension."""
    return Path(file).stem + ".jpg"


def clusters_table(index: pd.DataFrame, labels: np.ndarray, filename_names: np.ndarray, pred: np.ndarray,
                   conf: np.ndarray, conf_names: np.ndarray, pos: np.ndarray) -> pd.DataFrame:
    """The per-photo table behind every figure."""
    return pd.DataFrame({
        "file": index.file.to_numpy(),
        "thumb": [thumb_name(f) for f in index.file],
        "provenance": index.provenance.to_numpy(),
        "cluster": labels.astype(int),
        "filename_name": filename_names,
        "pred_name": pred,
        "pred_conf": conf,
        "confident_name": conf_names,
        "fa2_x": pos[:, 0],
        "fa2_y": pos[:, 1],
    })
