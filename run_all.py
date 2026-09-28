"""Run the whole hidden-stratification audit pipeline end to end.

    python run_all.py --root /path/to/SD                 # full profile (about 22 min on 1 CPU thread)
    python run_all.py --root /path/to/SD --profile quick # the 6 fast candidates (about 3.5 min)

Outputs (in outputs/): clusters.csv, validity.csv, choice.md, crosstab_catalogue.csv, split_scores.csv,
orphans.csv, stability.csv, stability_pairs.csv, run_log.txt, run_log.json, cache/ (refit labels).
Human annotation labels are not read anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import limit_threads  # noqa: E402  (must run before numpy is imported)

limit_threads(1)

import numpy as np  # noqa: E402

from src import data, pipeline, utils  # noqa: E402

OUTPUT_TABLES = ["clusters.csv", "validity.csv", "choice.md", "crosstab_catalogue.csv", "split_scores.csv",
                 "orphans.csv", "stability.csv", "stability_pairs.csv"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.yaml (default: next to this script)")
    ap.add_argument("--root", default=None, help="project root SD/ (overrides SD_ROOT and config)")
    ap.add_argument("--out", default=None, help="output folder (default: outputs/ next to this script)")
    ap.add_argument("--profile", choices=["quick", "full"], default=None, help="default: config 'profile'")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write outputs/cache/")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = utils.load_config(args.config)
    paths = utils.resolve_paths(cfg, args.root, args.out)
    profile = args.profile or cfg["profile"]
    out = paths["outputs"]
    out.mkdir(parents=True, exist_ok=True)
    log = utils.get_logger(out / "run_log.txt")
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    versions = utils.library_versions()
    inputs = {k: {"path": str(paths[k]), "sha256": utils.sha256(paths[k])} for k in ("embeddings", "index")}
    log.info(f"hidden-stratification audit pipeline, profile={profile}, config={cfg['_config_path']}")
    log.info("versions: " + ", ".join(f"{k}={v}" for k, v in versions.items()))
    for k, v in inputs.items():
        log.info(f"input {k}: {v['path']} sha256={v['sha256']}")

    with utils.timed("1_data", timings, log):
        ds = data.load_dataset(paths["embeddings"], paths["index"], cfg)
        features = data.build_features(ds.E, cfg)
    log.info(f"rows kept: {ds.n} (catalogue {ds.catalogue.sum()}, scraped {ds.scraped.sum()})")

    with utils.timed("2_candidates", timings, log):
        data_id = inputs["embeddings"]["sha256"] + inputs["index"]["sha256"]
        cache = None if args.no_cache else paths["cache"]
        cand = pipeline.evaluate_candidates(features, cfg, profile, data_id, cache, log)
    timings.update({f"2_candidates/{k}": v for k, v in cand.timings.items()})

    with utils.timed("3_selection", timings, log):
        key, table, md = pipeline.run_selection(cand.table, cfg, cand.skipped)
        utils.write_csv(table, out / "validity.csv", cfg)
        (out / "choice.md").write_text(md, encoding="utf-8")
    log.info(f"label-free choice: {key}")

    with utils.timed("4_audit", timings, log):
        labels = cand.refits[key].labels
        aud = pipeline.run_audit(ds, labels, cand.refits[key], cand.subsamples, cfg)
        utils.write_csv(aud.crosstab_catalogue, out / "crosstab_catalogue.csv", cfg)
        utils.write_csv(aud.split_scores, out / "split_scores.csv", cfg)
        utils.write_csv(aud.orphans, out / "orphans.csv", cfg)
        utils.write_csv(aud.stability, out / "stability.csv", cfg)
        utils.write_csv(aud.stability_pairs, out / "stability_pairs.csv", cfg)
    orphans = aud.orphans.loc[aud.orphans.orphan, "cluster"].tolist()
    split_names = aud.split_scores.loc[aud.split_scores.split & aud.split_scores.view.str.startswith("catalogue"),
                                       "name"].tolist()
    log.info(f"orphans (confident majority name < {cfg['audit']['orphan_max_share']:.0%} of the cluster): {orphans}")
    log.info(f"names split over >= 2 clusters (catalogue view): {split_names}")

    with utils.timed("5_export", timings, log):
        clusters, pos_source = pipeline.run_export(ds, labels, aud, cfg, paths["reference"])
        utils.write_csv(clusters, out / "clusters.csv", cfg)
    log.info(f"FA2 layout: {pos_source}")

    timings["total"] = time.perf_counter() - t_start
    digests = {name: utils.sha256(out / name) for name in OUTPUT_TABLES}
    for name, digest in digests.items():
        log.info(f"output {name}: sha256={digest}")
    log.info(f"[time] total: {timings['total']:.1f} s")
    run_info = {"profile": profile, "chosen": key, "n_rows": ds.n, "versions": versions, "inputs": inputs,
                "timings_s": {k: round(v, 2) for k, v in timings.items()}, "outputs_sha256": digests,
                "n_clusters": int(len(np.unique(labels)))}
    (out / "run_log.json").write_text(json.dumps(run_info, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
