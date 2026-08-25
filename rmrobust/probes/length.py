"""Probe 1 -- length bias, and the headline decomposition.

The question this probe exists to answer: *how much of the reward model's preference
is explained by response length alone?* Four increasingly demanding answers, because
the cheap ones are the ones usually reported and they overstate the case in opposite
directions.

1. **Comparative.** How well does "prefer the longer response" do on its own
   (`acc_longer`), and how well does a calibrated one-feature logistic on
   delta-log-length do (`acc_length_lr`)? If the RM scores 0.68 and length scores 0.62,
   the honest headline is not "the RM is 68% accurate".

2. **Correlational.** How much of the *variance in reward* is a monotone function of
   length (`r2_isotonic`), and what is the reward slope per 100 tokens? This is about
   the score, not the decision, and it is the number that matters for RLHF: a policy
   optimising this reward reads the slope, not the accuracy.

3. **Decompositional (the headline).** Fit a monotone g(length) out-of-fold, subtract
   it, and re-run the comparison on the residual reward. Then

       length_explained_fraction = (acc - acc_residual) / (acc - 0.5)

   -- the share of the RM's *above-chance* accuracy that disappears when you remove
   everything a monotone function of length could have provided. This is the number to
   quote. It is a fraction of skill, not of variance, and unlike a correlation it is
   comparable across models with different reward scales.

4. **Stratified.** Accuracy restricted to pairs the length heuristic cannot help with
   (`acc_length_neutral`), and accuracy as a function of the length gap. A model whose
   accuracy on length-neutral pairs is 0.5 has learned length and nothing else, however
   good its headline accuracy looks.

Caveats built into the output rather than left to the reader:
* Every metric is also computed with truncated pairs dropped, because under a token
  limit the longer response is the one that got cut.
* Everything is computed per source subset. In HH the sign of the length bias is not
  constant across subsets, so a pooled number can average two real effects to nothing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import stats as S
from ..features import log_len
from ..scoring import ScoredPairs

FEATURE_KEYS = (
    "n_chars", "n_words", "n_sentences", "n_bullets", "mean_word_len",
    "type_token_ratio", "n_question_marks", "has_hedge", "has_agreement",
    "has_refusal", "uppercase_frac",
)


def _delta_log_len(sp: ScoredPairs, unit: str) -> np.ndarray:
    a, b = sp.lengths(unit)
    return np.array([log_len(x) - log_len(y) for x, y in zip(a, b)])


def _surface_delta_matrix(sp: ScoredPairs) -> np.ndarray:
    rows = []
    for fc, fr in zip(sp.feat_chosen, sp.feat_rejected):
        rows.append([getattr(fc, k) - getattr(fr, k) for k in FEATURE_KEYS])
    return np.asarray(rows, dtype=float)


def _crossfit_residual_margin(sp: ScoredPairs, unit: str, n_folds: int = 2, seed: int = 0) -> np.ndarray:
    """Out-of-fold residual margin: r(x) - g_hat(len(x)) for both sides of every pair,
    with g_hat fit on the other folds' responses.

    Cross-fitting matters. Fitting g on the same responses you then residualize lets the
    isotonic regression absorb genuine reward signal that merely happens to correlate
    with length in that sample, and the length-explained fraction comes out too high.
    """
    lc, lr = sp.lengths(unit)
    x = np.concatenate([[log_len(v) for v in lc], [log_len(v) for v in lr]])
    y = np.concatenate([sp.r_chosen, sp.r_rejected])
    n = len(sp)

    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, size=n)
    fold = np.concatenate([fold, fold])  # both sides of a pair share a fold

    resid = np.zeros_like(y)
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        if tr.sum() < 10 or te.sum() == 0:
            resid[te] = y[te] - y[tr].mean() if tr.sum() else y[te]
            continue
        g = S.IsotonicLengthModel().fit(x[tr], y[tr])
        resid[te] = y[te] - g.predict(x[te])
    return resid[:n] - resid[n:]


def _core(sp: ScoredPairs, unit: str = "tokens", seed: int = 0, n_boot: int = 2000) -> Dict:
    if len(sp) < 20:
        return {"n": len(sp), "insufficient_data": True}

    correct = sp.correct
    dl = _delta_log_len(sp, unit)
    lc, lr = sp.lengths(unit)

    # --- 1. comparative baselines ------------------------------------------------
    longer_correct = np.where(dl > 0, 1.0, np.where(dl < 0, 0.0, 0.5))

    # A calibrated length baseline has to be symmetrised: every pair contributes
    # (+delta, label=1) and (-delta, label=0). Without that, a logistic on delta can
    # cheat with an intercept and "predict chosen" is trivially 100% accurate.
    len_lr_correct = S.cross_val_accuracy(
        np.concatenate([dl, -dl]).reshape(-1, 1),
        np.concatenate([np.ones(len(dl)), np.zeros(len(dl))]),
        seed=seed,
    )[: len(dl)]
    # Same trick over all surface features: an upper bound on what is reachable
    # without reading the response.
    surf = _surface_delta_matrix(sp)
    surf_lr_correct = S.cross_val_accuracy(
        np.concatenate([surf, -surf], axis=0),
        np.concatenate([np.ones(len(surf)), np.zeros(len(surf))]),
        seed=seed,
    )[: len(surf)]

    # --- 2. correlational --------------------------------------------------------
    all_len = np.concatenate([lc, lr])
    all_r = np.concatenate([sp.r_chosen, sp.r_rejected])
    all_loglen = np.array([log_len(v) for v in all_len])
    iso = S.IsotonicLengthModel().fit(all_loglen, all_r)
    slope = float(np.polyfit(all_len, all_r, 1)[0]) if all_len.std() > 0 else float("nan")
    reward_sd = float(all_r.std())

    # --- 3. decomposition --------------------------------------------------------
    resid_margin = _crossfit_residual_margin(sp, unit, seed=seed)
    resid_correct = np.where(resid_margin > 0, 1.0, np.where(resid_margin < 0, 0.0, 0.5))

    acc = S.bootstrap(correct, n_boot=n_boot, seed=seed)
    acc_resid = S.bootstrap(resid_correct, n_boot=n_boot, seed=seed)
    drop = S.paired_bootstrap_diff(correct, resid_correct, n_boot=n_boot, seed=seed)

    def _frac(c, rc):
        a, ar = float(np.mean(c)), float(np.mean(rc))
        return (a - ar) / (a - 0.5) if abs(a - 0.5) > 1e-9 else float("nan")

    rngb = np.random.default_rng(seed + 1)
    idx = rngb.integers(0, len(correct), size=(n_boot, len(correct)))
    frac_boot = np.array([_frac(correct[i], resid_correct[i]) for i in idx])
    frac_boot = frac_boot[np.isfinite(frac_boot)]
    # The fraction has (acc - 0.5) in its denominator, so it explodes when the model is
    # near chance. Winsorise the bootstrap for reporting and flag the estimate as
    # unreliable unless the accuracy CI itself clears chance -- a ratio whose denominator
    # is indistinguishable from zero is not a measurement.
    if len(frac_boot):
        frac_boot = np.clip(frac_boot, -3.0, 3.0)
        frac_lo, frac_hi = np.percentile(frac_boot, [2.5, 97.5])
    else:
        frac_lo = frac_hi = np.nan
    frac_reliable = bool(acc.lo > 0.5)

    # --- 4. stratified -----------------------------------------------------------
    neutral = np.abs(dl) <= 0.05          # within ~5% relative length
    tight = np.abs(lc - lr) <= 2          # within 2 tokens/words/chars
    strata = []
    if len(dl) >= 50:
        edges = np.percentile(dl, np.linspace(0, 100, 6))
        edges[0], edges[-1] = -np.inf, np.inf
        for i in range(5):
            m = (dl >= edges[i]) & (dl < edges[i + 1])
            if m.sum() >= 10:
                strata.append({
                    "bin": i,
                    "delta_log_len_lo": float(edges[i]) if np.isfinite(edges[i]) else None,
                    "delta_log_len_hi": float(edges[i + 1]) if np.isfinite(edges[i + 1]) else None,
                    "n": int(m.sum()),
                    "acc": float(correct[m].mean()),
                    "acc_residual": float(resid_correct[m].mean()),
                    "acc_longer_heuristic": float(longer_correct[m].mean()),
                })

    # --- 5. agreement structure --------------------------------------------------
    rm_pick_longer = np.where(sp.margin > 0, dl > 0, dl < 0)
    both = (correct > 0.5) & (longer_correct > 0.5)
    rm_only = (correct > 0.5) & (longer_correct < 0.5)
    len_only = (correct < 0.5) & (longer_correct > 0.5)
    neither = (correct < 0.5) & (longer_correct < 0.5)

    return {
        "n": len(sp),
        "unit": unit,
        "reward_sd": reward_sd,
        "mean_len_chosen": float(lc.mean()),
        "mean_len_rejected": float(lr.mean()),
        "frac_chosen_longer": float((dl > 0).mean()),
        "comparative": {
            "acc_rm": acc.as_dict(),
            "acc_longer_heuristic": S.bootstrap(longer_correct, n_boot=n_boot, seed=seed).as_dict(),
            "acc_length_logistic_oof": S.bootstrap(len_lr_correct, n_boot=n_boot, seed=seed).as_dict(),
            "acc_surface_logistic_oof": S.bootstrap(surf_lr_correct, n_boot=n_boot, seed=seed).as_dict(),
            "rm_minus_length_logistic": S.paired_bootstrap_diff(correct, len_lr_correct, n_boot=n_boot, seed=seed).as_dict(),
            "rm_minus_surface_logistic": S.paired_bootstrap_diff(correct, surf_lr_correct, n_boot=n_boot, seed=seed).as_dict(),
        },
        "correlational": {
            "pearson_reward_vs_loglen": S.pearson(all_loglen, all_r),
            "spearman_reward_vs_len": S.spearman(all_len, all_r),
            "r2_isotonic_reward_on_loglen": iso.r2(all_loglen, all_r),
            "reward_slope_per_100_units": slope * 100.0,
            "reward_slope_per_100_units_in_sd": (slope * 100.0 / reward_sd) if reward_sd > 0 else float("nan"),
            "pearson_margin_vs_delta_loglen": S.pearson(dl, sp.margin),
        },
        "decomposition": {
            "acc_rm": acc.as_dict(),
            "acc_residual_after_length": acc_resid.as_dict(),
            "acc_drop_paired": drop.as_dict(),
            "length_explained_fraction": {
                "value": _frac(correct, resid_correct),
                "ci_lo": float(frac_lo),
                "ci_hi": float(frac_hi),
                "n": len(correct),
                "reliable": frac_reliable,
                "note": ("undefined near chance: the denominator is (acc - 0.5). "
                         "reliable=false means the accuracy CI includes chance, so this "
                         "ratio should not be quoted. CI is winsorised to [-3, 3]."),
            },
        },
        "stratified": {
            "acc_length_neutral_5pct": S.bootstrap(correct[neutral], n_boot=n_boot, seed=seed).as_dict() if neutral.sum() >= 20 else {"n": int(neutral.sum())},
            "acc_length_tight_2units": S.bootstrap(correct[tight], n_boot=n_boot, seed=seed).as_dict() if tight.sum() >= 20 else {"n": int(tight.sum())},
            "bins": strata,
        },
        "agreement": {
            "rm_picks_longer_frac": float(np.mean(rm_pick_longer)),
            "agreement_with_length_heuristic": float(np.mean((sp.margin > 0) == (dl > 0))),
            "contingency": {
                "rm_right_len_right": int(both.sum()),
                "rm_right_len_wrong": int(rm_only.sum()),
                "rm_wrong_len_right": int(len_only.sum()),
                "rm_wrong_len_wrong": int(neither.sum()),
            },
            "acc_given_length_heuristic_right": float(correct[longer_correct > 0.5].mean()) if (longer_correct > 0.5).sum() else float("nan"),
            "acc_given_length_heuristic_wrong": float(correct[longer_correct < 0.5].mean()) if (longer_correct < 0.5).sum() else float("nan"),
        },
    }


def run(
    sp: ScoredPairs,
    units: Sequence[str] = ("tokens", "chars"),
    per_source: bool = True,
    seed: int = 0,
    n_boot: int = 2000,
) -> Dict:
    out: Dict = {"probe": "length", "overall": {}, "by_source": {}, "truncation": {}}

    for u in units:
        out["overall"][u] = _core(sp, unit=u, seed=seed, n_boot=n_boot)

    trunc = sp.truncated_mask()
    out["truncation"] = {
        "frac_pairs_with_truncated_response": float(trunc.mean()),
        "untruncated_only": _core(sp.subset(~trunc), unit=units[0], seed=seed, n_boot=n_boot)
        if (~trunc).sum() >= 20 else {"n": int((~trunc).sum())},
    }

    if per_source:
        for src in sorted(set(sp.sources.tolist())):
            m = sp.sources == src
            if m.sum() >= 20:
                out["by_source"][src] = _core(sp.subset(m), unit=units[0], seed=seed, n_boot=n_boot)

    return out
