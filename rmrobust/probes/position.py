"""Probe 3 -- position bias.

"Position bias" in the RLHF literature usually means the A/B ordering effect in an
LLM-as-judge: swap the two candidates and the judge changes its mind. A pointwise
reward model scores one response at a time, so that specific effect does not exist for
it. What does exist, and matters just as much for what RLHF optimises, is *where in the
text* the model's score comes from. Three measurements:

1. **segment_swap.** A response built from two independent segments, concatenated in
   both orders. The two versions contain the same tokens. Any reward difference is
   position and nothing else -- not length, not content, not style. On the
   order-invariant items (two parallel tips, two parallel causes) the correct answer is
   exactly zero; the measured value is the model's raw positional sensitivity. On the
   answer/caveat items the signed effect says whether the model pays for leading with
   the answer or leading with the hedge.

2. **constraint_depth.** A user constraint is stated in an early turn, the question comes
   last, and bland filler turns are inserted between them to push the constraint further
   back. The reward gap between a compliant and a violating response, as a function of
   that distance, is how deep into the context the model is actually reading. Because it
   is a difference of differences, the response-length gap cancels: whatever confound
   exists at distance 0 exists identically at distance 3.

3. **prefix_truncation.** Real HH responses truncated to their first k tokens. Two
   readings: how well the score from k tokens correlates with the score from the whole
   response, and how much of the model's *pairwise accuracy* survives on the prefix
   alone. If accuracy at 32 tokens matches accuracy at 512, the model is grading the
   opening and skimming the rest -- which is precisely what a policy optimising against
   it will learn to exploit.

4. **sentence_reversal.** A control for the first measurement. Real HH responses with
   their sentences reversed. A model that shows zero effect here is order-blind (a bag
   of words with extra steps), which makes a null in `segment_swap` uninformative rather
   than reassuring.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import stats as S
from ..data import Pair
from ..model import RewardModelBase

DEFAULT_PROBESET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "probesets", "position_v1.json",
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def load_probeset(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_PROBESET, encoding="utf-8") as f:
        return json.load(f)


def _ctx(user_text: str) -> str:
    return f"\n\nHuman: {user_text}\n\nAssistant:"


def _ctx_turns(turns: Sequence[Tuple[str, str]]) -> str:
    return "".join(f"\n\n{r}: {t}" for r, t in turns) + "\n\nAssistant:"


# --------------------------------------------------------------------------------------


def _segment_swap(model, doc, reward_sd, batch_size, n_boot, seed, progress) -> Dict:
    items = [i for i in doc["items"] if i["arm"] == "segment_swap"]
    if not items:
        return {"n": 0}
    ctxs, resps = [], []
    for it in items:
        c = _ctx(it["question"])
        ctxs += [c, c]
        resps += [it["segment_1"] + " " + it["segment_2"], it["segment_2"] + " " + it["segment_1"]]
    s = model.score(ctxs, resps, batch_size=batch_size, progress=progress)
    d = s[0::2] - s[1::2]  # reward(segment 1 first) - reward(segment 2 first)

    inv = np.array([bool(i["order_invariant"]) for i in items])
    topics = [i["topic"] for i in items]

    def block(mask, signed_name):
        if mask.sum() == 0:
            return {"n": 0}
        v = d[mask]
        est = S.cluster_bootstrap(v, [t for t, m in zip(topics, mask) if m], n_boot=n_boot, seed=seed)
        absest = S.cluster_bootstrap(np.abs(v), [t for t, m in zip(topics, mask) if m], n_boot=n_boot, seed=seed)
        return {
            "n_items": int(mask.sum()),
            signed_name: est.as_dict(),
            "signed_effect_in_reward_sd": est.value / reward_sd if reward_sd > 0 else float("nan"),
            "mean_abs_effect": absest.as_dict(),
            "mean_abs_effect_in_reward_sd": absest.value / reward_sd if reward_sd > 0 else float("nan"),
            "frac_prefers_first_ordering": float(np.mean(v > 0)),
        }

    out = {
        "order_invariant_items": block(inv, "signed_effect_segment1_first_minus_segment2_first"),
        "answer_vs_caveat_items": block(~inv, "signed_effect_answer_first_minus_caveat_first"),
        "note": "identical token multiset in both orderings; any non-zero value is position alone",
    }
    out["order_invariant_items"]["interpretation"] = (
        "signed effect should be 0 for an order-neutral model; mean_abs_effect is the "
        "raw positional sensitivity and is the number to compare against reward sd"
    )
    return out


def _constraint_depth(model, doc, reward_sd, batch_size, n_boot, seed, progress,
                      depths: Sequence[int] = (0, 1, 2, 3)) -> Dict:
    items = [i for i in doc["items"] if i["arm"] == "constraint_depth"]
    if not items:
        return {"n": 0}
    filler = [tuple(t) for t in doc["filler_turns"]]

    ctxs, resps, tags = [], [], []
    for it in items:
        for d in depths:
            turns: List[Tuple[str, str]] = [("Human", it["constraint"]),
                                            ("Assistant", "Understood, I'll keep that in mind.")]
            turns += filler[: 2 * d]
            turns += [("Human", it["question"])]
            c = _ctx_turns(turns)
            for key in ("response_compliant", "response_violating"):
                ctxs.append(c)
                resps.append(it[key])
                tags.append((it["topic"], d, key))
    s = model.score(ctxs, resps, batch_size=batch_size, progress=progress)
    lut = {t: v for t, v in zip(tags, s)}

    by_depth = {}
    gaps_by_depth = {}
    for d in depths:
        gaps, tops = [], []
        for it in items:
            g = lut[(it["topic"], d, "response_compliant")] - lut[(it["topic"], d, "response_violating")]
            gaps.append(g)
            tops.append(it["topic"])
        gaps = np.asarray(gaps)
        gaps_by_depth[d] = gaps
        est = S.cluster_bootstrap(gaps, tops, n_boot=n_boot, seed=seed)
        by_depth[str(d)] = {
            "n_filler_turn_pairs": d,
            "compliance_gap": est.as_dict(),
            "compliance_gap_in_reward_sd": est.value / reward_sd if reward_sd > 0 else float("nan"),
            "frac_compliant_preferred": float(np.mean(gaps > 0)),
        }

    dd = np.repeat(np.asarray(depths, float), len(items))
    gg = np.concatenate([gaps_by_depth[d] for d in depths])
    reg = S.ols_intercept_slope(dd, gg, n_boot=n_boot, seed=seed)
    g0, gN = gaps_by_depth[depths[0]], gaps_by_depth[depths[-1]]
    decay = S.paired_bootstrap_diff(g0, gN, n_boot=n_boot, seed=seed)

    return {
        "n_items": len(items),
        "depths": list(depths),
        "by_depth": by_depth,
        "gap_vs_depth_regression": reg,
        "gap_decay_shallow_minus_deep": decay.as_dict(),
        "retention_ratio_deep_over_shallow": (
            float(gN.mean() / g0.mean()) if abs(g0.mean()) > 1e-9 else float("nan")
        ),
        "note": "difference of differences: the response-length gap is identical at every depth and cancels",
    }


def _prefix_truncation(model, pairs: Sequence[Pair], reward_sd, batch_size, n_boot, seed,
                       progress, ks: Sequence[int] = (8, 16, 32, 64, 128)) -> Dict:
    if len(pairs) < 20:
        return {"n": len(pairs)}
    ctx = [p.context_text() for p in pairs]
    full_c = model.score(ctx, [p.chosen for p in pairs], batch_size=batch_size, progress=progress)
    full_r = model.score(ctx, [p.rejected for p in pairs], batch_size=batch_size, progress=progress)
    full_correct = ((full_c - full_r) > 0).astype(float)
    full_acc = float(full_correct.mean())

    rows = []
    for k in ks:
        ch = [model.truncate_to_tokens(p.chosen, k) for p in pairs]
        rj = [model.truncate_to_tokens(p.rejected, k) for p in pairs]
        sc = model.score(ctx, ch, batch_size=batch_size, progress=False)
        sr = model.score(ctx, rj, batch_size=batch_size, progress=False)
        correct = ((sc - sr) > 0).astype(float)
        acc = S.bootstrap(correct, n_boot=n_boot, seed=seed)
        agree = float(np.mean(((sc - sr) > 0) == ((full_c - full_r) > 0)))
        rows.append({
            "k_tokens": int(k),
            "acc_from_prefix": acc.as_dict(),
            "frac_of_full_skill_recovered": (
                (acc.value - 0.5) / (full_acc - 0.5) if abs(full_acc - 0.5) > 1e-9 else float("nan")
            ),
            "decision_agreement_with_full": agree,
            "pearson_reward_prefix_vs_full": S.pearson(np.concatenate([sc, sr]),
                                                       np.concatenate([full_c, full_r])),
            "mean_abs_reward_shift_in_sd": (
                float(np.mean(np.abs(np.concatenate([sc - full_c, sr - full_r])))) / reward_sd
                if reward_sd > 0 else float("nan")
            ),
        })
    return {
        "n_pairs": len(pairs),
        "acc_full_response": full_acc,
        "by_prefix_length": rows,
        "note": "frac_of_full_skill_recovered near 1.0 at small k means the model grades the opening",
    }


def _sentence_reversal(model, pairs: Sequence[Pair], reward_sd, batch_size, n_boot, seed,
                       progress, min_sentences: int = 4) -> Dict:
    cand = []
    for p in pairs:
        sents = [s for s in _SENT_SPLIT.split(p.chosen.strip()) if s.strip()]
        if len(sents) >= min_sentences:
            cand.append((p, sents))
    if len(cand) < 20:
        return {"n": len(cand), "note": "too few multi-sentence responses"}
    ctx = [p.context_text() for p, _ in cand]
    orig = [p.chosen for p, _ in cand]
    rev = [" ".join(reversed(s)) for _, s in cand]
    so = model.score(ctx, orig, batch_size=batch_size, progress=progress)
    sr = model.score(ctx, rev, batch_size=batch_size, progress=False)
    d = so - sr
    est = S.bootstrap(d, n_boot=n_boot, seed=seed)
    return {
        "n_responses": len(cand),
        "mean_reward_original_minus_reversed": est.as_dict(),
        "in_reward_sd": est.value / reward_sd if reward_sd > 0 else float("nan"),
        "mean_abs_shift_in_sd": float(np.mean(np.abs(d))) / reward_sd if reward_sd > 0 else float("nan"),
        "frac_original_preferred": float(np.mean(d > 0)),
        "note": ("a value near zero means the model is order-blind, which makes a null in "
                 "segment_swap uninformative rather than reassuring"),
    }


def run(
    model: RewardModelBase,
    pairs: Optional[Sequence[Pair]] = None,
    probeset_path: Optional[str] = None,
    reward_sd: float = 1.0,
    batch_size: int = 16,
    n_boot: int = 2000,
    seed: int = 0,
    max_hh_pairs: int = 500,
    progress: bool = True,
) -> Dict:
    doc = load_probeset(probeset_path)
    out: Dict = {
        "probe": "position",
        "probeset_version": doc.get("version"),
        "reward_sd_reference": reward_sd,
        "segment_swap": _segment_swap(model, doc, reward_sd, batch_size, n_boot, seed, progress),
        "constraint_depth": _constraint_depth(model, doc, reward_sd, batch_size, n_boot, seed, progress),
    }
    if pairs:
        sub = list(pairs)[:max_hh_pairs]
        out["prefix_truncation"] = _prefix_truncation(model, sub, reward_sd, batch_size, n_boot, seed, progress)
        out["sentence_reversal"] = _sentence_reversal(model, sub, reward_sd, batch_size, n_boot, seed, progress)
    return out
