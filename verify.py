"""Check the outputs of run_all.py against the artefacts of the original analysis (reference/).

    python verify.py            # reads outputs/ and reference/, writes outputs/verification.txt

Checks
1. the chosen labelling equals reference/raw__Leiden_r1.0.npy (ARI and contingency table)
2. the label-free choice equals the original choice (reference/choice_r3.md)
3. validity numbers equal the original internal.csv for every candidate that was run
4. the kNN name vote equals the original reference/scraped_pred_device.csv
5. the FA2 positions equal reference/pos_raw.npy
6. refit stability equals the original split_stability.py output (3 decimals)
Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
METRICS = ["n_clusters", "noise_frac", "largest_frac", "sil", "db", "ch", "stab_sub_ari", "stab_seed_ari"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--reference", default=str(HERE / "reference"))
    args = ap.parse_args()
    out, ref = Path(args.out), Path(args.reference)
    lines, ok = [], {}

    clusters = pd.read_csv(out / "clusters.csv", keep_default_na=False)
    lab = clusters.cluster.to_numpy()
    ref_lab = np.load(ref / "raw__Leiden_r1.0.npy")
    ari = adjusted_rand_score(ref_lab, lab)
    ct = pd.crosstab(pd.Series(ref_lab, name="reference"), pd.Series(lab, name="pipeline"))
    ok["1 labels identical"] = bool(np.array_equal(ref_lab, lab))
    lines += [f"1. chosen labelling vs reference: ARI = {ari:.6f}, identical = {ok['1 labels identical']}, "
              f"n = {len(lab)}, clusters = {lab.max() + 1} vs {ref_lab.max() + 1}",
              f"   contingency: {int((ct.values > 0).sum())} non-zero cells, diagonal holds "
              f"{int(np.trace(ct.values)) if ct.shape[0] == ct.shape[1] else 'n/a'} of {len(lab)} photos"]
    ct.to_csv(out / "verification_contingency.csv", lineterminator="\n")

    validity = pd.read_csv(out / "validity.csv")
    chosen = validity.loc[validity.chosen, "key"].item()
    ref_choice = re.search(r"\*\*Chosen:\*\* `([^`]+)`", (ref / "choice_r3.md").read_text()).group(1)
    ok["2 same choice"] = chosen == ref_choice
    lines.append(f"2. label-free choice: pipeline {chosen}, original {ref_choice}, same = {ok['2 same choice']}")

    internal = pd.read_csv(ref / "internal_r3.csv").set_index("key")
    comp = validity[validity.source == "computed"].set_index("key")
    diffs = (comp[METRICS] - internal.loc[comp.index, METRICS]).abs()
    ok["3 validity equal"] = bool(diffs.max().max() < 1e-6)  # csv keeps 6 decimals
    lines.append(f"3. validity vs original internal.csv on {len(comp)} computed candidates: "
                 f"max |diff| = {diffs.max().max():.2e} (csv keeps 6 decimals), equal = {ok['3 validity equal']}")
    for key, row in diffs.iterrows():
        lines.append(f"   {key:22s} max |diff| {row.max():.2e}")

    pv = pd.read_csv(ref / "scraped_pred_device.csv").set_index("file")
    scr = clusters[clusters.provenance == "scraped"].set_index("file")
    same_pred = (scr.pred_name == pv.loc[scr.index, "pred"]).all()
    conf_diff = np.abs(scr.pred_conf.astype(float) - pv.loc[scr.index, "conf"]).max()
    ok["4 name vote equal"] = bool(same_pred and conf_diff < 1e-6)
    lines.append(f"4. kNN name vote on {len(scr)} scraped photos: same names = {same_pred}, "
                 f"max |conf diff| = {conf_diff:.1e}")

    pos = clusters[["fa2_x", "fa2_y"]].to_numpy(float)
    ref_pos = np.load(ref / "pos_raw.npy")
    pos_diff = np.abs(pos - ref_pos).max()
    ok["5 layout equal"] = bool(pos_diff < 1e-5)
    lines.append(f"5. FA2 positions: max |diff| = {pos_diff:.1e} (csv keeps 6 decimals)")

    st = pd.read_csv(out / "stability.csv").set_index("cluster")
    rs = pd.read_csv(ref / "split_stability_r3.csv").set_index("cluster")
    cols = {"mean_top_share": "mean_top_share", "min_top_share": "min_top_share",
            "frac_refits_hold": "frac_refits_ge_0.60"}
    sd = max(np.abs(st.loc[rs.index, a].round(3) - rs[b]).max() for a, b in cols.items())
    pairs = pd.read_csv(out / "stability_pairs.csv").set_index("pair")
    rp = pd.read_csv(ref / "split_stability_pairs_r3.csv").set_index("pair")
    pd_diff = np.abs(pairs.loc[rp.index, "frac_refits_apart"].round(2) - rp.frac_refits_apart).max()
    ok["6 stability equal"] = bool(sd < 1e-9 and pd_diff < 1e-9)
    lines.append(f"6. refit stability vs original split_stability.py: max |diff| groups {sd:.3f}, pairs {pd_diff:.2f}")

    lines.append("")
    lines += [f"{'PASS' if v else 'FAIL'}  {k}" for k, v in ok.items()]
    text = "\n".join(lines) + "\n"
    (out / "verification.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
