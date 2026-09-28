"""Robustness check: GEORGE Step 1 subclass estimation per coarse name (see src/george.py).

Run after run_all.py (it reads outputs/clusters.csv for the Leiden labels and the name vote):
    python george_check.py --root /path/to/SD
Writes outputs/george_step1.csv: for each coarse name and variant, the k chosen by silhouette, the number of
subclasses after over-clustering, the ARI with the Leiden clusters, and where each subclass lands among the
Leiden clusters (majority cluster, its share, subclass size).
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

from src import data, george, utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--config"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    utils.limit_threads(1)
    cfg = utils.load_config(a.config)
    paths = utils.resolve_paths(cfg, a.root)
    ds = data.load_dataset(paths["embeddings"], paths["index"], cfg)
    Z = data.deflate(ds.E, 0)
    C = pd.read_csv(paths["outputs"] / "clusters.csv", keep_default_na=False)
    assert (C.file.values == ds.index.file.values).all(), "clusters.csv is not aligned with the dataset"
    L = C.cluster.to_numpy()
    views = {"filename name (catalogue)": C.filename_name.to_numpy(),
             "confident name vote (all photos)": C.confident_name.to_numpy()}
    rows = []
    for view, names in views.items():
        for algo in ("bit", "umap"):
            for n in sorted(set(names) - {""}):
                m = np.where(names == n)[0]
                X = george.reduce(Z[m], algo, a.seed)
                k, sc, lab = george.estimate_subclasses(X, algo, a.seed)
                land = []
                for c in np.unique(lab):
                    v = pd.Series(L[m][lab == c]).value_counts()
                    land.append(f"{int(v.index[0])}:{v.iloc[0] / v.sum():.2f}(n={v.sum()})")
                rows.append(dict(view=view, variant=algo, name=n, n=len(m), k_silhouette=k, silhouette=round(sc, 3),
                                 n_subclasses=int(lab.max() + 1), ari_vs_leiden=round(adjusted_rand_score(L[m], lab), 3),
                                 subclass_to_leiden=" ".join(land)))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(paths["outputs"] / "george_step1.csv", index=False)
    cat = C.filename_name.to_numpy() != ""
    h, c, v = homogeneity_completeness_v_measure(C.filename_name[cat], L[cat])
    pd.DataFrame([dict(homogeneity=round(h, 4), completeness=round(c, 4), v_measure=round(v, 4))]).to_csv(
        paths["outputs"] / "v_measure_catalogue.csv", index=False)
    print(f"Leiden vs filename names (catalogue): homogeneity {h:.3f}, completeness {c:.3f}, V {v:.3f}")


if __name__ == "__main__":
    main()
