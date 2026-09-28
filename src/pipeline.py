"""The four stages of the pipeline, shared by run_all.py and notebook.ipynb.

1. candidates  fit every candidate configuration with its stability refits        [label-free]
2. selection   apply the label-free choice rule                                   [label-free]
3. audit       describe the chosen clusters with filename names                   [post hoc]
4. export      per-photo table with FA2 positions for the figures
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import audit, cluster, data, figures, select, validity

VALIDITY_COLUMNS = ["key", "features", "method", "source", "n_clusters", "noise_frac", "largest_frac", "sil", "db",
                    "ch", "stab_sub_ari", "stab_sub_sd", "n_sub", "stab_seed_ari", "stab_seed_sd",
                    "n_seed_labellings"]


@dataclass
class CandidateResults:
    table: pd.DataFrame
    refits: dict[str, validity.Refits]
    subsamples: np.ndarray
    skipped: list[str]
    timings: dict[str, float] = field(default_factory=dict)


def candidates_for(cfg: dict, profile: str) -> tuple[list[dict], list[str]]:
    """Candidates to fit in this profile, and the keys skipped by it."""
    if profile not in ("quick", "full"):
        raise ValueError("profile must be 'quick' or 'full'")
    run = [c for c in cfg["candidates"] if profile == "full" or c["profile"] == "quick"]
    skipped = [c["key"] for c in cfg["candidates"] if c not in run]
    return run, skipped


def evaluate_candidates(features: dict[str, np.ndarray], cfg: dict, profile: str, data_id: str,
                        cache_dir: Path | None, logger: logging.Logger) -> CandidateResults:
    """Stage 1: fit each candidate (reported labelling + 20 subsample refits + 5 seed refits) and score it."""
    st = cfg["stability"]
    n = len(next(iter(features.values())))
    subsamples = validity.make_subsamples(n, st["n_subsamples"], st["fraction"], st["rng_seed"])
    specs, skipped = candidates_for(cfg, profile)
    rows, refits, timings = [], {}, {}
    for spec in specs:
        t0 = time.perf_counter()
        Z = features[spec["features"]]
        fp = validity.fingerprint(spec, cfg, subsamples, data_id)
        cache = cache_dir / f"{spec['key']}.npz" if cache_dir else None
        refits[spec["key"]] = validity.fit_refits(cluster.make_method(spec, cfg), Z, subsamples, cfg, cache, fp)
        row = {"key": spec["key"], "features": spec["features"], "method": cluster.method_label(spec),
               "source": "computed", **validity.validity_row(Z, refits[spec["key"]], subsamples)}
        rows.append(row)
        timings[spec["key"]] = time.perf_counter() - t0
        logger.info(f"{spec['key']:22s} k={row['n_clusters']:3d} sil={row['sil']:.3f} "
                    f"resample ARI={row['stab_sub_ari']:.3f} seed ARI={row['stab_seed_ari']:.3f} "
                    f"({timings[spec['key']]:.0f} s)")
    for rec in cfg.get("recorded", []):
        rows.append({**rec, "source": "recorded (original TEMI run, not re-fitted)", "noise_frac": 0.0})
    table = pd.DataFrame(rows).reindex(columns=VALIDITY_COLUMNS)
    if skipped:
        logger.info(f"not run in the '{profile}' profile: {', '.join(skipped)}")
    return CandidateResults(table, refits, subsamples, skipped, timings)


def run_selection(table: pd.DataFrame, cfg: dict, skipped: list[str]) -> tuple[str, pd.DataFrame, str]:
    """Stage 2: the label-free choice. Returns the key, the annotated table and the markdown record."""
    sc = cfg["selection"]
    key, annotated = select.choose(table, sc["min_clusters"], sc["tie_margin"])
    md = select.choice_markdown(key, annotated, sc["min_clusters"], sc["tie_margin"])
    if skipped:
        md += f"\nNot run in this profile (use --profile full): {', '.join(skipped)}.\n"
    return key, annotated, md


@dataclass
class AuditResults:
    filename_names: np.ndarray
    pred: np.ndarray
    conf: np.ndarray
    confident: np.ndarray
    crosstab_catalogue: pd.DataFrame
    split_scores: pd.DataFrame
    orphans: pd.DataFrame
    stability: pd.DataFrame
    stability_pairs: pd.DataFrame


def run_audit(ds: data.Dataset, labels: np.ndarray, refits: validity.Refits, subsamples: np.ndarray,
              cfg: dict) -> AuditResults:
    """Stage 3: post-hoc description with filename names (catalogue) and the kNN name vote (scraped)."""
    ac = cfg["audit"]
    cat, scr = ds.catalogue, ds.scraped
    fnames = ds.index[ac["name_column"]].fillna("").to_numpy(dtype=object)
    fnames[~cat] = ""
    pred, conf = audit.name_vote(data.l2_uncentred_float32(ds.E), fnames, cat, scr, ac["name_knn_k"])
    confident = audit.confident_names(fnames, pred, conf, cat, ac["confident_conf"])
    orphans = audit.orphan_table(labels, confident, cat, ac["orphan_max_share"])
    majority = audit.majority_map(orphans)
    splits = pd.concat([
        audit.split_scores(labels, fnames, cat, ac["split_min_share"], "catalogue filename name", majority),
        audit.split_scores(labels, confident, np.ones(len(labels), bool), ac["split_min_share"],
                           "confident name (catalogue + scraped vote)", majority),
    ], ignore_index=True)
    stab, pairs = audit.refit_stability(labels, refits, subsamples, ac["groups"], ac["pairs"], ac["hold_share"])
    ct = audit.crosstab(labels, fnames, cat & (fnames != "")).reset_index()
    return AuditResults(fnames, pred, conf, confident, ct, splits, orphans, stab, pairs)


def run_export(ds: data.Dataset, labels: np.ndarray, aud: AuditResults, cfg: dict,
               reference_dir: Path) -> tuple[pd.DataFrame, str]:
    """Stage 4: the per-photo table with ForceAtlas2 positions (drawn on the raw kNN graph)."""
    pos, source = figures.load_layout(data.centred_l2_float32(ds.E), cfg, reference_dir)
    table = figures.clusters_table(ds.index, labels, aud.filename_names, aud.pred, aud.conf, aud.confident, pos)
    return table, source
