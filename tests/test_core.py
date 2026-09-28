"""Small unit tests on toy data (no input files needed).

    python tests/test_core.py        # or: python -m pytest tests/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import audit, data, select, validity  # noqa: E402


def test_deflate_is_centred_and_unit_norm():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(50, 8)) + 3.0
    for n_pc in (0, 2):
        Z = data.deflate(E, n_pc)
        assert np.allclose(np.linalg.norm(Z, axis=1), 1.0)
    Zc = E - E.mean(0)
    assert np.allclose(data.deflate(E, 0), Zc / np.linalg.norm(Zc, axis=1, keepdims=True))


def test_subsamples_are_reproducible_and_sorted():
    a = validity.make_subsamples(100, 4, 0.8, 12345)
    b = validity.make_subsamples(100, 4, 0.8, 12345)
    assert a.shape == (4, 80) and np.array_equal(a, b)
    assert all(np.all(np.diff(row) > 0) for row in a)


def test_mean_pair_ari_on_shared_points():
    idx = [np.array([0, 1, 2, 3]), np.array([2, 3, 4, 5])]
    labs = [np.array([0, 0, 1, 1]), np.array([5, 5, 7, 7])]   # shared rows 2, 3 carry labels (1, 1) and (5, 5)
    mean, sd = validity.mean_pair_ari(labs, idx)
    assert mean == 1.0 and sd == 0.0


def test_choice_rule_uses_tie_margin_then_silhouette():
    t = pd.DataFrame({"key": ["a", "b", "c", "d"], "features": "raw", "method": "m", "source": "computed",
                      "n_clusters": [12, 15, 8, 20], "stab_sub_ari": [0.905, 0.900, 0.99, np.nan],
                      "stab_seed_ari": 0.9, "sil": [0.10, 0.20, 0.5, 0.9]})
    key, out = select.choose(t, min_clusters=10, tie_margin=0.01)
    assert key == "b"                                   # c has < 10 clusters, d has no estimate
    assert out.set_index("key").eligible.to_dict() == {"a": True, "b": True, "c": False, "d": False}


def test_split_score_and_orphan_rule():
    labels = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    names = np.array(["TV", "TV", "TV", "TV", "x", "", "", ""], dtype=object)
    s = audit.split_scores(labels, names, np.ones(8, bool), 0.10, "toy").set_index("name")
    assert np.isclose(s.loc["TV", "split_score"], np.log(2) / np.log(3)) and bool(s.loc["TV", "split"])
    o = audit.orphan_table(labels, names, np.zeros(8, bool), 0.50).set_index("cluster")
    assert o.orphan.to_dict() == {0: False, 1: False, 2: True}     # cluster 2: "x" covers 1 of 4


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print("ok", f.__name__)
