#!/usr/bin/env python3
"""Training-free baselines on the HH-RLHF test splits.

Every number here is available before a reward model exists, and every one of them is a
number the reward model has to beat to be doing anything a regular expression could not.
Run this first; it takes about a minute and it sets the bar the rest of the study is
measured against.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rmrobust import data as D
from rmrobust import stats as S
from rmrobust.features import featurize, log_len
from rmrobust.probes.length import FEATURE_KEYS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/day0_length_baselines.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    out = {"split": a.split, "by_subset": {}}
    pooled = {"dl_words": [], "dl_chars": [], "surf": []}

    for sub in D.SUBSETS:
        pairs = D.load_pairs(a.data_dir, sub, a.split)
        fc = [featurize(p.chosen) for p in pairs]
        fr = [featurize(p.rejected) for p in pairs]
        dl_w = np.array([log_len(x.n_words) - log_len(y.n_words) for x, y in zip(fc, fr)])
        dl_c = np.array([log_len(x.n_chars) - log_len(y.n_chars) for x, y in zip(fc, fr)])
        surf = np.array([[getattr(x, k) - getattr(y, k) for k in FEATURE_KEYS] for x, y in zip(fc, fr)])

        def sym_acc(X, seed=0):
            X = np.asarray(X, float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            XX = np.concatenate([X, -X], axis=0)
            yy = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
            return S.cross_val_accuracy(XX, yy, seed=seed)[: len(X)]

        longer_w = np.where(dl_w > 0, 1.0, np.where(dl_w < 0, 0.0, 0.5))
        longer_c = np.where(dl_c > 0, 1.0, np.where(dl_c < 0, 0.0, 0.5))
        neutral = np.abs(dl_w) <= 0.05

        out["by_subset"][sub] = {
            "n_pairs": len(pairs),
            "mean_words_chosen": float(np.mean([f.n_words for f in fc])),
            "mean_words_rejected": float(np.mean([f.n_words for f in fr])),
            "frac_chosen_longer_words": float((dl_w > 0).mean()),
            "median_delta_log_words": float(np.median(dl_w)),
            "acc_longer_wins_words": S.bootstrap(longer_w, n_boot=a.n_boot).as_dict(),
            "acc_longer_wins_chars": S.bootstrap(longer_c, n_boot=a.n_boot).as_dict(),
            "acc_logistic_on_delta_log_words": S.bootstrap(sym_acc(dl_w), n_boot=a.n_boot).as_dict(),
            "acc_logistic_on_all_surface_features": S.bootstrap(sym_acc(surf), n_boot=a.n_boot).as_dict(),
            "frac_length_neutral_within_5pct": float(neutral.mean()),
            "n_length_neutral": int(neutral.sum()),
        }
        pooled["dl_words"].append(dl_w)
        pooled["surf"].append(surf)

        r = out["by_subset"][sub]
        print(f"{sub:28s} n={len(pairs):5d}  chosen/rejected words "
              f"{r['mean_words_chosen']:6.1f}/{r['mean_words_rejected']:6.1f}  "
              f"longer-wins {r['acc_longer_wins_words']['value']:.3f}  "
              f"logistic(len) {r['acc_logistic_on_delta_log_words']['value']:.3f}  "
              f"logistic(surface) {r['acc_logistic_on_all_surface_features']['value']:.3f}  "
              f"length-neutral {r['frac_length_neutral_within_5pct']:.1%}")

    dl = np.concatenate(pooled["dl_words"])
    surf = np.concatenate(pooled["surf"], axis=0)

    def sym_acc(X, seed=0):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        XX = np.concatenate([X, -X], axis=0)
        yy = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
        return S.cross_val_accuracy(XX, yy, seed=seed)[: len(X)]

    longer = np.where(dl > 0, 1.0, np.where(dl < 0, 0.0, 0.5))
    out["pooled_all_subsets"] = {
        "n_pairs": int(len(dl)),
        "acc_longer_wins_words": S.bootstrap(longer, n_boot=a.n_boot).as_dict(),
        "acc_logistic_on_delta_log_words": S.bootstrap(sym_acc(dl), n_boot=a.n_boot).as_dict(),
        "acc_logistic_on_all_surface_features": S.bootstrap(sym_acc(surf), n_boot=a.n_boot).as_dict(),
    }
    p = out["pooled_all_subsets"]
    print(f"\n{'POOLED':28s} n={p['n_pairs']:5d}  longer-wins {p['acc_longer_wins_words']['value']:.3f}  "
          f"logistic(len) {p['acc_logistic_on_delta_log_words']['value']:.3f}  "
          f"logistic(surface) {p['acc_logistic_on_all_surface_features']['value']:.3f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
