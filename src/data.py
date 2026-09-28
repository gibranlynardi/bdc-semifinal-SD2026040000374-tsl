"""Load the frozen DINOv3 embeddings and build the label-free feature variants.

Input: ``emb_full_A144.npy`` (26,527 x 768, float16) and ``emb_full_index.csv`` (same row order).
We keep the rows of class ``1_Electronic`` with ``keep == 1`` (one photo per perceptual-hash duplicate
group), which gives 3,607 photos: 2,332 catalogue photos and 1,275 scraped (field) photos.
Only the index columns ``file``, ``cls``, ``provenance``, ``device_label`` and ``keep`` are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

INDEX_COLUMNS = ["file", "cls", "provenance", "device_label", "keep"]


@dataclass
class Dataset:
    """The analysed photos. ``E`` is float64, rows aligned with ``index``."""

    E: np.ndarray
    index: pd.DataFrame

    @property
    def n(self) -> int:
        return len(self.index)

    @property
    def catalogue(self) -> np.ndarray:
        return (self.index.provenance == "catalogue").to_numpy()

    @property
    def scraped(self) -> np.ndarray:
        return (self.index.provenance == "scraped").to_numpy()


def load_dataset(embeddings: Path, index: Path, cfg: dict) -> Dataset:
    """Select the kept Electronic rows. The float16 -> float64 cast is exact."""
    idx = pd.read_csv(index, usecols=INDEX_COLUMNS)
    X = np.load(embeddings)
    if len(idx) != len(X):
        raise ValueError(f"index has {len(idx)} rows but embeddings have {len(X)}")
    dc = cfg["data"]
    mask = ((idx.cls == dc["class_name"]) & (idx[dc["keep_column"]] == 1)).to_numpy()
    E = X[mask].astype(np.float64)
    I = idx[mask].reset_index(drop=True)
    if len(I) != dc["expected_rows"]:
        raise ValueError(f"expected {dc['expected_rows']} kept rows, got {len(I)}")
    if I.file.duplicated().any():
        raise ValueError("duplicate file names among kept rows")
    return Dataset(E=E, index=I)


def deflate(E: np.ndarray, n_components: int, random_state: int = 0) -> np.ndarray:
    """Mean-centre, optionally project out the top principal components, then L2-normalise each row.

    ``n_components = 0`` gives the "raw" features used by the chosen configuration.
    Identical to ``base.deflate`` of the original analysis.
    """
    Zc = E - E.mean(0)
    if n_components:
        V = PCA(n_components=n_components, random_state=random_state).fit(Zc).components_
        Zc = Zc - (Zc @ V.T) @ V
    return Zc / np.linalg.norm(Zc, axis=1, keepdims=True)


def build_features(E: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """All feature variants named in ``cfg['features']`` (name -> number of PCs projected out)."""
    return {name: deflate(E, n_pc, cfg["pca_random_state"]) for name, n_pc in cfg["features"].items()}


def l2_uncentred_float32(E: np.ndarray) -> np.ndarray:
    """Row-L2-normalised float32 embeddings without centring (used by the post-hoc name vote)."""
    Z = E.astype(np.float32)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def centred_l2_float32(E: np.ndarray) -> np.ndarray:
    """Mean-centred, L2-normalised features computed in float32 (used by the ForceAtlas2 layout only).

    The original layout was computed in float32. Its kNN edge weights differ from the float64 route by
    at most about 2e-5, which does not change the clustering but does change the layout coordinates.
    """
    Z = E.astype(np.float32)
    Zc = Z - Z.mean(0)
    return Zc / np.linalg.norm(Zc, axis=1, keepdims=True)
