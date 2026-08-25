"""Ground-truth tests: inject a known effect, assert the probe reads it back."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rmrobust import data as D
from rmrobust.features import featurize
from rmrobust.probes import length as P_length
from rmrobust.probes import position as P_position
from rmrobust.probes import shift as P_shift
from rmrobust.probes import sycophancy as P_syc
from rmrobust.scoring import ScoredPairs, score_pairs

from scripted import (ScriptedRewardModel, bag_of_words, constraint_aware,
                      first_k_words, length_only, oracle, recent_context_only)

DATA_DIR = "data"
N = 500


@pytest.fixture(scope="module")
def pairs():
    return D.load_pairs(DATA_DIR, "helpful-base", "test", limit=N)


def _scored(pairs, fn, noise=0.0, seed=0):
    m = ScriptedRewardModel(fn, noise=noise, seed=seed)
    return score_pairs(m, pairs, progress=False), m


# ---------------------------------------------------------------- length probe


def test_length_probe_recovers_a_pure_length_model(pairs):
    sp, _ = _scored(pairs, length_only(3.0), noise=0.15)
    r = P_length.run(sp, units=("words",), per_source=False, n_boot=500)["overall"]["words"]
    lef = r["decomposition"]["length_explained_fraction"]
    assert lef["ci_lo"] <= 1.0 <= lef["ci_hi"], f"CI {lef} should cover 1.0"
    assert lef["value"] > 0.6
    assert r["correlational"]["r2_isotonic_reward_on_loglen"] > 0.9
    # residualising a pure length model must take it to chance
    assert abs(r["decomposition"]["acc_residual_after_length"]["value"] - 0.5) < 0.05


def test_length_probe_null_on_a_content_only_model(pairs):
    sp, _ = _scored(pairs, oracle([p.chosen for p in pairs], bonus=1.0), noise=0.2)
    r = P_length.run(sp, units=("words",), per_source=False, n_boot=500)["overall"]["words"]
    lef = r["decomposition"]["length_explained_fraction"]
    assert abs(lef["value"]) < 0.15, f"content-only model should show ~0 length share, got {lef}"
    assert r["decomposition"]["acc_rm"]["value"] > 0.85
    assert r["correlational"]["r2_isotonic_reward_on_loglen"] < 0.1


def test_length_probe_flags_unreliable_near_chance(pairs):
    sp, _ = _scored(pairs, lambda c, r: 0.0, noise=1.0)
    r = P_length.run(sp, units=("words",), per_source=False, n_boot=400)["overall"]["words"]
    assert r["decomposition"]["length_explained_fraction"]["reliable"] is False


def test_length_baseline_matches_pure_length_model_accuracy(pairs):
    """A model that IS length should score exactly what the length baseline scores."""
    sp, _ = _scored(pairs, length_only(50.0))
    r = P_length.run(sp, units=("words",), per_source=False, n_boot=300)["overall"]["words"]
    a = r["comparative"]["acc_rm"]["value"]
    b = r["comparative"]["acc_longer_heuristic"]["value"]
    assert abs(a - b) < 0.02, (a, b)


# ---------------------------------------------------------------- sycophancy


def _sycophantic_scorer(delta: float, include_placebo: bool = False):
    doc = P_syc.load_probeset()
    bonus = {}
    for it in doc["items"]:
        if it["arm"] not in ("stance_flip", "placebo"):
            continue
        if it["arm"] == "placebo" and not include_placebo:
            continue
        fr = doc["framings"]["stance" if it["arm"] == "stance_flip" else "placebo"][it["framing"]]
        ca = P_syc._ctx(fr.format(stance=it["stance_a"]) + " " + it["question"])
        cb = P_syc._ctx(fr.format(stance=it["stance_b"]) + " " + it["question"])
        bonus[(ca, it["response_a"])] = delta
        bonus[(cb, it["response_b"])] = delta

    def fn(c, r):
        base = 0.01 * len(r.split())
        return base + bonus.get((c, r), 0.0)

    return fn


def test_sycophancy_recovers_injected_effect():
    delta = 0.4
    m = ScriptedRewardModel(_sycophantic_scorer(delta), noise=0.02)
    res = P_syc.run(m, reward_sd=1.0, n_boot=400, progress=False)
    sf = res["stance_flip"]["sycophancy_reward"]
    assert abs(sf["value"] - delta) < 0.05, sf
    assert sf["ci_lo"] <= delta <= sf["ci_hi"]
    # placebo arm was not given the bonus, so it must read ~0
    assert abs(res["placebo"]["sycophancy_reward"]["value"]) < 0.05
    assert res["stance_flip"]["echo_win_rate_per_response"] > 0.95


def test_sycophancy_null_for_a_context_blind_model():
    m = ScriptedRewardModel(lambda c, r: 0.01 * len(r.split()), noise=0.0)
    res = P_syc.run(m, reward_sd=1.0, n_boot=300, progress=False)
    assert abs(res["stance_flip"]["sycophancy_reward"]["value"]) < 1e-9
    assert abs(res["placebo"]["sycophancy_reward"]["value"]) < 1e-9


def test_sycophancy_placebo_catches_a_spurious_context_effect():
    """A model that rewards long contexts, not agreement, must show up in the placebo."""
    fn = _sycophantic_scorer(0.3, include_placebo=True)
    m = ScriptedRewardModel(fn, noise=0.01)
    res = P_syc.run(m, reward_sd=1.0, n_boot=300, progress=False)
    assert res["placebo"]["sycophancy_reward"]["value"] > 0.2
    net = res["stance_minus_placebo"]["value"]
    assert abs(net) < 0.1, "placebo subtraction should cancel a non-specific context effect"


# ---------------------------------------------------------------- position


def test_segment_swap_is_exactly_zero_for_an_order_blind_model():
    m = ScriptedRewardModel(bag_of_words(), noise=0.0)
    res = P_position.run(m, pairs=None, reward_sd=1.0, n_boot=200, progress=False)
    inv = res["segment_swap"]["order_invariant_items"]
    assert abs(inv["mean_abs_effect"]["value"]) < 1e-9
    assert abs(inv["signed_effect_segment1_first_minus_segment2_first"]["value"]) < 1e-9


def test_segment_swap_detects_an_opening_reader():
    m = ScriptedRewardModel(first_k_words(10), noise=0.0)
    res = P_position.run(m, pairs=None, reward_sd=1.0, n_boot=200, progress=False)
    inv = res["segment_swap"]["order_invariant_items"]
    assert inv["mean_abs_effect"]["value"] > 0.05, inv


def test_constraint_gap_decays_when_the_model_only_reads_recent_context():
    """Ground truth: a model with a hard 260-character attention horizon. It should show
    the full compliance gap at depth 0 and none of it once filler pushes the constraint
    outside that window."""
    m = ScriptedRewardModel(constraint_aware(window_chars=260, bonus=1.0), noise=0.0)
    res = P_position.run(m, pairs=None, reward_sd=1.0, n_boot=200, progress=False)
    by = res["constraint_depth"]["by_depth"]
    shallow = by["0"]["compliance_gap"]["value"]
    deep = by["3"]["compliance_gap"]["value"]
    assert shallow > 0.9, shallow
    assert deep < 0.1, deep
    assert res["constraint_depth"]["gap_vs_depth_regression"]["slope"] < 0
    assert res["constraint_depth"]["retention_ratio_deep_over_shallow"] < 0.15


def test_constraint_gap_is_flat_for_a_model_that_reads_the_whole_context():
    m = ScriptedRewardModel(constraint_aware(window_chars=100000, bonus=1.0), noise=0.0)
    res = P_position.run(m, pairs=None, reward_sd=1.0, n_boot=200, progress=False)
    by = res["constraint_depth"]["by_depth"]
    assert abs(by["0"]["compliance_gap"]["value"] - by["3"]["compliance_gap"]["value"]) < 1e-9
    assert abs(res["constraint_depth"]["retention_ratio_deep_over_shallow"] - 1.0) < 1e-9


def test_prefix_probe_saturates_for_an_opening_reader(pairs):
    m = ScriptedRewardModel(first_k_words(8), noise=0.0)
    res = P_position.run(m, pairs=pairs[:200], reward_sd=1.0, n_boot=200,
                         max_hh_pairs=200, progress=False)
    rows = {r["k_tokens"]: r for r in res["prefix_truncation"]["by_prefix_length"]}
    assert rows[16]["decision_agreement_with_full"] > 0.98
    assert rows[8]["decision_agreement_with_full"] <= rows[32]["decision_agreement_with_full"]


def test_sentence_reversal_null_for_an_order_blind_model(pairs):
    m = ScriptedRewardModel(bag_of_words(), noise=0.0)
    res = P_position.run(m, pairs=pairs[:300], reward_sd=1.0, n_boot=200,
                         max_hh_pairs=300, progress=False)
    rev = res["sentence_reversal"]
    if rev.get("n_responses"):
        assert rev["mean_abs_shift_in_sd"] < 1e-6


# ---------------------------------------------------------------- shift


def test_perturbations_never_flip_an_invariant_model(pairs):
    """A model that ignores the response entirely cannot have its decision flipped."""
    m = ScriptedRewardModel(oracle([p.chosen for p in pairs], bonus=1.0), noise=0.0)
    # oracle keys on exact text, so a rewrite destroys the match; use a truly invariant fn
    m2 = ScriptedRewardModel(lambda c, r: float(len(c)), noise=0.0)
    res = P_shift.surface_perturbation(m2, pairs[:120], reward_sd=1.0, n_boot=200, progress=False)
    for name, r in res["by_perturbation"].items():
        assert r["decision_flip_rate_both_sided"] == 0.0, name
        assert abs(r["reward_shift_in_sd"]) < 1e-9, name


def test_perturbation_reward_shift_tracks_added_length(pairs):
    m = ScriptedRewardModel(length_only(1.0), noise=0.0)
    res = P_shift.surface_perturbation(m, pairs[:120], reward_sd=1.0, n_boot=200, progress=False)
    by = res["by_perturbation"]
    assert by["appended_disclaimer"]["reward_shift_one_sided"]["value"] > 0
    assert by["enthusiastic_opener"]["reward_shift_one_sided"]["value"] > 0
    assert abs(by["lowercase"]["reward_shift_one_sided"]["value"]) < 1e-9


def test_cross_source_finds_the_domain_it_was_told_about(pairs):
    by_src = {s: D.load_pairs(DATA_DIR, s, "test", limit=200)
              for s in ("helpful-base", "harmless-base")}
    # a model that prefers the longer response: strong on helpful, inverted on harmless
    m = ScriptedRewardModel(length_only(3.0), noise=0.05)
    res = P_shift.cross_source(m, by_src, reference_source="helpful-base", n_boot=300, progress=False)
    acc_help = res["by_source"]["helpful-base"]["acc"]["value"]
    acc_harm = res["by_source"]["harmless-base"]["acc"]["value"]
    assert acc_help > acc_harm, (acc_help, acc_harm)
    assert res["by_source"]["harmless-base"]["acc_drop_vs_reference"] > 0


def test_calibration_is_symmetrised():
    """Without symmetrisation every label is 1 and ECE is meaningless."""
    margins = np.full(400, 3.0)  # model always right, always confident
    cal = P_shift._calibration(margins)
    assert abs(cal["empirical_rate"] - 0.5) < 1e-9
    assert cal["ece"] < 0.1
