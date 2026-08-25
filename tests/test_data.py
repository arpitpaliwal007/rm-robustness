"""Parsing and sampling tests against the real dataset."""

from __future__ import annotations

import numpy as np
import pytest

from rmrobust import data as D
from rmrobust.train import length_balanced_subset

DATA_DIR = "data"


def test_all_subsets_parse():
    for sub in D.SUBSETS:
        raw = list(D.iter_raw(DATA_DIR, sub, "test"))
        parsed = D.load_pairs(DATA_DIR, sub, "test", drop_degenerate=False)
        assert len(parsed) == len(raw), f"{sub}: {len(parsed)} parsed of {len(raw)}"


def test_context_plus_response_reconstructs_the_transcript():
    """The split must be lossless: whatever we call context plus whatever we call the
    response has to be the original string, or every length statistic is wrong."""
    n_checked = 0
    for rec in list(D.iter_raw(DATA_DIR, "helpful-base", "test"))[:400]:
        p = D.parse_pair(rec, "u", "helpful-base", "test")
        if p is None or p.diverged_early:
            continue
        rebuilt = p.context_text() + " " + p.chosen
        assert rebuilt.split() == rec["chosen"].split(), rebuilt[:200]
        rebuilt_r = p.context_text() + " " + p.rejected
        assert rebuilt_r.split() == rec["rejected"].split()
        n_checked += 1
    assert n_checked > 300


def test_responses_exclude_the_shared_context():
    pairs = D.load_pairs(DATA_DIR, "helpful-base", "test", limit=200)
    for p in pairs[:50]:
        assert "\n\nHuman:" not in p.chosen or p.diverged_early
        assert p.chosen != p.rejected


def test_helpful_and_harmless_disagree_about_length():
    """The property the whole distribution-shift story rests on."""
    means = {}
    for sub in ("helpful-base", "harmless-base"):
        ps = D.load_pairs(DATA_DIR, sub, "test")
        means[sub] = np.mean([len(p.chosen) - len(p.rejected) for p in ps])
    assert means["helpful-base"] > 0, means
    assert means["harmless-base"] < 0, means


def test_length_balanced_subset_is_actually_balanced():
    ps = D.load_pairs(DATA_DIR, "helpful-base", "train", limit=4000)
    bal = length_balanced_subset(ps, unit="chars", seed=0)
    d = np.array([len(p.chosen) - len(p.rejected) for p in bal])
    assert len(bal) > 1000
    assert abs(np.mean(d > 0) - 0.5) < 1e-9
    # and the magnitudes are matched, not just the signs
    assert abs(np.mean(np.abs(d[d > 0])) - np.mean(np.abs(d[d < 0]))) / np.mean(np.abs(d)) < 0.15


def test_train_val_split_is_disjoint_and_seeded():
    ps = D.load_pairs(DATA_DIR, "helpful-base", "train", limit=1000)
    a1, b1 = D.train_val_split(ps, 0.1, seed=7)
    a2, b2 = D.train_val_split(ps, 0.1, seed=7)
    assert [p.uid for p in b1] == [p.uid for p in b2]
    assert not (set(p.uid for p in a1) & set(p.uid for p in b1))
    assert len(a1) + len(b1) == len(ps)
