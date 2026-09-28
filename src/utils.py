"""Configuration, paths, logging and provenance helpers."""
from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent.parent

# Distributions whose versions are written to the run log and pinned in requirements.txt.
TRACKED_LIBRARIES = (
    "numpy", "pandas", "scipy", "scikit-learn", "scanpy", "anndata", "igraph", "leidenalg",
    "umap-learn", "pynndescent", "numba", "fa2-modified", "PyYAML", "tabulate",
)


def limit_threads(n: int = 1) -> None:
    """Pin BLAS / OpenMP / numba thread pools. Must run before numpy is imported to take full effect."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, str(n))


def load_config(path: str | Path | None = None) -> dict:
    """Read config.yaml (default: the one next to run_all.py)."""
    path = Path(path) if path else PACKAGE_DIR / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def resolve_paths(cfg: dict, root: str | None = None, out_dir: str | None = None) -> dict:
    """Return absolute paths. Root precedence: argument, SD_ROOT env var, config (relative to the config file)."""
    base = Path(cfg["_config_path"]).parent
    root = root or os.environ.get("SD_ROOT") or cfg["paths"]["root"]
    root = Path(root)
    if not root.is_absolute():
        root = (base / root).resolve()
    out = Path(out_dir) if out_dir else base / cfg["paths"]["outputs"]
    paths = {
        "root": root,
        "embeddings": root / cfg["paths"]["embeddings"],
        "index": root / cfg["paths"]["index"],
        "outputs": out.resolve(),
        "cache": out.resolve() / "cache",
        "reference": (base / cfg["paths"]["reference"]).resolve(),
    }
    for key in ("embeddings", "index"):
        if not paths[key].exists():
            raise FileNotFoundError(f"{key} not found at {paths[key]}. Pass --root or set SD_ROOT.")
    return paths


def get_logger(log_file: Path | None = None) -> logging.Logger:
    """Logger that writes to stdout and, optionally, to a run log file."""
    logger = logging.getLogger("hstrat")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        fileh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fileh.setFormatter(fmt)
        logger.addHandler(fileh)
    return logger


@contextmanager
def timed(label: str, timings: dict, logger: logging.Logger | None = None) -> Iterator[None]:
    """Record the wall time of a block in ``timings[label]`` (seconds)."""
    t0 = time.perf_counter()
    yield
    timings[label] = time.perf_counter() - t0
    if logger:
        logger.info(f"[time] {label}: {timings[label]:.1f} s")


def library_versions() -> dict:
    """Installed versions of the tracked libraries (``None`` when not installed)."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in TRACKED_LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        out[var] = os.environ.get(var)
    return out


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, to pin the exact input in the run log."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(df, path: Path, cfg: dict) -> None:
    """Write a CSV with a fixed float format so reruns are byte-identical."""
    df.to_csv(path, index=False, float_format=cfg["output"]["float_format"], lineterminator="\n")
