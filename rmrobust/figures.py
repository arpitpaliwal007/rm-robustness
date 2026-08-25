"""Figures for the robustness report.

Design rules applied throughout: one y-axis per chart (never two scales), categorical
hues assigned in fixed order and never cycled, at most three series per chart, recessive
grid and axes, direct labels rather than a number on every point, and error bars on
every estimate that has them. Charts render light-mode PNG at 2x for embedding.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed order, never cycled
MUTED = "#a8a7a2"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10.5,
    "axes.titleweight": "600",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
})


def _clean(ax, xgrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y")
    if xgrid:
        ax.grid(axis="y", visible=False)
    else:
        ax.grid(axis="x", visible=False)


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _v(d, *keys, default=float("nan")):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d if not isinstance(d, dict) else d.get("value", default)


# --------------------------------------------------------------------------------------


def length_decomposition(res: Dict, out_dir: str, unit: str = "tokens") -> Optional[str]:
    """The headline chart: what each successively weaker predictor achieves."""
    core = res.get("overall", {}).get(unit)
    if not core or core.get("insufficient_data"):
        return None
    c, d = core["comparative"], core["decomposition"]
    rows = [
        ("Reward model", c["acc_rm"]),
        ("RM, length removed", d["acc_residual_after_length"]),
        ("Surface features only", c["acc_surface_logistic_oof"]),
        ("Length only", c["acc_length_logistic_oof"]),
        ("Longer wins", c["acc_longer_heuristic"]),
    ]
    labels = [r[0] for r in rows]
    vals = np.array([r[1].get("value", np.nan) for r in rows])
    lo = np.array([r[1].get("ci_lo", np.nan) for r in rows])
    hi = np.array([r[1].get("ci_hi", np.nan) for r in rows])
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.barh(y, vals - 0.5, left=0.5, height=0.58, color=SERIES[0], zorder=3)
    ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="none",
                ecolor=INK_2, elinewidth=1.2, capsize=3, zorder=4)
    ax.axvline(0.5, color=MUTED, lw=1.2, zorder=2)
    hi_x = max(np.nanmax(hi), 0.55)
    for yi, h, v in zip(y, hi, vals):
        ax.text(h + 0.004, yi, f"{v:.3f}", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    lo_x = min(0.5, np.nanmin(lo)) - 0.01
    ax.set_xlim(lo_x, hi_x + 0.10 * (hi_x - lo_x) + 0.03)
    ax.set_xlabel("pairwise accuracy on held-out HH (0.5 = chance)")
    ax.set_title("How much of the preference is length?", loc="left")
    _clean(ax, xgrid=True)
    return _save(fig, out_dir, "fig_length_decomposition.png")


def reward_vs_length(sp, out_dir: str, unit: str = "tokens", n_bins: int = 12) -> Optional[str]:
    """Mean reward as a function of response length, pooled over both sides."""
    lc, lr = sp.lengths(unit)
    lens = np.concatenate([lc, lr])
    rew = np.concatenate([sp.r_chosen, sp.r_rejected])
    if len(lens) < 100:
        return None
    sd = rew.std() or 1.0
    z = (rew - rew.mean()) / sd
    edges = np.unique(np.percentile(lens, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 4:
        return None
    xs, ys, es = [], [], []
    for i in range(len(edges) - 1):
        m = (lens >= edges[i]) & (lens < edges[i + 1] if i < len(edges) - 2 else lens <= edges[i + 1])
        if m.sum() >= 20:
            xs.append(float(np.median(lens[m])))
            ys.append(float(z[m].mean()))
            es.append(float(z[m].std() / np.sqrt(m.sum())))
    if len(xs) < 3:
        return None
    xs, ys, es = np.array(xs), np.array(ys), np.array(es)

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.fill_between(xs, ys - es, ys + es, color=SERIES[0], alpha=0.16, lw=0)
    ax.plot(xs, ys, color=SERIES[0], lw=2, marker="o", ms=5, zorder=3)
    ax.axhline(0, color=MUTED, lw=1.1)
    ax.annotate("mean reward", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(6, 0), va="center", color=SERIES[0], fontsize=8.5)
    ax.set_xlabel(f"response length ({unit})")
    ax.set_ylabel("reward (sd from mean)")
    ax.set_title("Reward against response length", loc="left")
    ax.set_xscale("log")
    _clean(ax)
    return _save(fig, out_dir, "fig_reward_vs_length.png")


def accuracy_by_length_gap(res: Dict, out_dir: str, unit: str = "tokens") -> Optional[str]:
    core = res.get("overall", {}).get(unit)
    if not core or not core.get("stratified", {}).get("bins"):
        return None
    bins = core["stratified"]["bins"]
    x = np.arange(len(bins))
    rm = np.array([b["acc"] for b in bins])
    rs = np.array([b["acc_residual"] for b in bins])
    labels = []
    for b in bins:
        lo, hi = b["delta_log_len_lo"], b["delta_log_len_hi"]
        labels.append(("<" + f"{hi:.2f}") if lo is None else (f">{lo:.2f}" if hi is None else f"{lo:.2f} to {hi:.2f}"))

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(x, rm, color=SERIES[0], lw=2, marker="o", ms=6, label="reward model", zorder=3)
    ax.plot(x, rs, color=SERIES[1], lw=2, marker="s", ms=6, label="length removed", zorder=3)
    ax.axhline(0.5, color=MUTED, lw=1.1)
    ax.annotate("reward model", (x[-1], rm[-1]), textcoords="offset points", xytext=(6, 0),
                va="center", color=SERIES[0], fontsize=8.5)
    ax.annotate("length removed", (x[-1], rs[-1]), textcoords="offset points", xytext=(6, 0),
                va="center", color=SERIES[1], fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("length gap, log(chosen) - log(rejected)")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by how much longer the preferred response is", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-0.4, len(bins) - 0.4 + 1.2)
    _clean(ax)
    return _save(fig, out_dir, "fig_accuracy_by_length_gap.png")


def probe_effects(results: Dict, out_dir: str) -> Optional[str]:
    """One dot per probe effect, in reward-sd units, with the zero line."""
    rows: List = []

    syc = results.get("sycophancy", {})
    if syc:
        sf = syc.get("stance_flip", {}).get("sycophancy_in_reward_sd")
        pl = syc.get("placebo", {}).get("sycophancy_in_reward_sd")
        if sf:
            rows.append(("Sycophancy: echo user's stance", sf))
        if pl:
            rows.append(("  placebo floor (irrelevant stance)", pl))
        cap = syc.get("capitulation", {}).get("delta_in_reward_sd")
        if cap:
            rows.append(("Capitulation over holding correct", cap))
        fl = syc.get("flattery", {}).get("delta_in_reward_sd")
        if fl:
            rows.append(("Praise over critique of user's work", fl))

    pos = results.get("position", {})
    inv = pos.get("segment_swap", {}).get("order_invariant_items", {})
    if inv.get("mean_abs_effect"):
        e = inv["mean_abs_effect"]
        sd = pos.get("reward_sd_reference", 1.0) or 1.0
        rows.append(("Position: |shift| from reordering same text",
                     {"value": e["value"] / sd, "ci_lo": e["ci_lo"] / sd, "ci_hi": e["ci_hi"] / sd}))
    ac = pos.get("segment_swap", {}).get("answer_vs_caveat_items", {})
    if ac.get("signed_effect_answer_first_minus_caveat_first"):
        e = ac["signed_effect_answer_first_minus_caveat_first"]
        sd = pos.get("reward_sd_reference", 1.0) or 1.0
        rows.append(("Answer first over caveat first",
                     {"value": e["value"] / sd, "ci_lo": e["ci_lo"] / sd, "ci_hi": e["ci_hi"] / sd}))

    if not rows:
        return None
    labels = [r[0] for r in rows]
    vals = np.array([r[1].get("value", np.nan) for r in rows])
    lo = np.array([r[1].get("ci_lo", np.nan) for r in rows])
    hi = np.array([r[1].get("ci_hi", np.nan) for r in rows])
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(6.8, 0.46 * len(rows) + 1.4))
    ax.hlines(y, lo, hi, color=INK_2, lw=1.3, zorder=3)
    ax.scatter(vals, y, s=46, color=SERIES[0], zorder=4, edgecolor=SURFACE, linewidth=1.2)
    ax.axvline(0, color=MUTED, lw=1.2, zorder=2)
    span = float(np.nanmax(hi) - np.nanmin(lo)) or 1.0
    for yi, v, h in zip(y, vals, hi):
        ax.text(h + 0.02 * span, yi, f"{v:+.2f}", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_xlim(np.nanmin(lo) - 0.06 * span, np.nanmax(hi) + 0.22 * span)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("effect size (reward sd)")
    ax.set_title("Probe effects, in units of the reward's own spread", loc="left")
    _clean(ax, xgrid=True)
    return _save(fig, out_dir, "fig_probe_effects.png")


def prefix_recovery(results: Dict, out_dir: str) -> Optional[str]:
    pt = results.get("position", {}).get("prefix_truncation", {})
    rows = pt.get("by_prefix_length")
    if not rows:
        return None
    x = np.array([r["k_tokens"] for r in rows], dtype=float)
    y = np.array([r["frac_of_full_skill_recovered"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.plot(x, y, color=SERIES[0], lw=2, marker="o", ms=6, zorder=3)
    ax.axhline(1.0, color=MUTED, lw=1.1)
    ax.annotate("full response", (x[-1], 1.0), textcoords="offset points", xytext=(4, 4),
                color=INK_2, fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("response tokens visible to the model")
    ax.set_ylabel("share of above-chance accuracy")
    ax.set_title("How much of the model's skill is in the opening tokens", loc="left")
    _clean(ax)
    return _save(fig, out_dir, "fig_prefix_recovery.png")


def constraint_depth(results: Dict, out_dir: str) -> Optional[str]:
    cd = results.get("position", {}).get("constraint_depth", {})
    by = cd.get("by_depth")
    if not by:
        return None
    ks = sorted(by, key=lambda k: int(k))
    x = np.array([int(k) for k in ks], dtype=float)
    g = np.array([by[k]["compliance_gap"]["value"] for k in ks])
    lo = np.array([by[k]["compliance_gap"]["ci_lo"] for k in ks])
    hi = np.array([by[k]["compliance_gap"]["ci_hi"] for k in ks])

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.fill_between(x, lo, hi, color=SERIES[0], alpha=0.16, lw=0)
    ax.plot(x, g, color=SERIES[0], lw=2, marker="o", ms=6, zorder=3)
    ax.axhline(0, color=MUTED, lw=1.1)
    ax.set_xticks(x)
    ax.set_xlabel("filler turn pairs between the constraint and the question")
    ax.set_ylabel("reward gap, compliant - violating")
    ax.set_title("Does the model still see a constraint stated earlier?", loc="left")
    _clean(ax)
    return _save(fig, out_dir, "fig_constraint_depth.png")


def cross_source_accuracy(results: Dict, out_dir: str) -> Optional[str]:
    cs = results.get("shift", {}).get("cross_source", {})
    by = cs.get("by_source")
    if not by:
        return None
    srcs = sorted(by)
    ref = cs.get("reference_source")
    vals = np.array([by[s]["acc"]["value"] for s in srcs])
    lo = np.array([by[s]["acc"]["ci_lo"] for s in srcs])
    hi = np.array([by[s]["acc"]["ci_hi"] for s in srcs])
    y = np.arange(len(srcs))[::-1]
    colors = [SERIES[0] if s == ref else SERIES[1] for s in srcs]

    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(srcs) + 1.5))
    ax.barh(y, vals - 0.5, left=0.5, height=0.55, color=colors, zorder=3)
    ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="none", ecolor=INK_2,
                elinewidth=1.2, capsize=3, zorder=4)
    ax.axvline(0.5, color=MUTED, lw=1.2)
    hi_x = max(np.nanmax(hi), 0.55)
    lo_x = min(0.5, np.nanmin(lo)) - 0.01
    for yi, h, v in zip(y, hi, vals):
        ax.text(h + 0.004, yi, f"{v:.3f}", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([s + ("  (trained here)" if s == ref else "") for s in srcs], fontsize=8.5)
    ax.set_xlim(lo_x, hi_x + 0.10 * (hi_x - lo_x) + 0.03)
    ax.set_xlabel("pairwise accuracy")
    ax.set_title("Accuracy across HH subsets", loc="left")
    _clean(ax, xgrid=True)
    return _save(fig, out_dir, "fig_cross_source_accuracy.png")


def cross_source_length_bias(results: Dict, out_dir: str) -> Optional[str]:
    """Deliberately a second chart rather than a second axis on the accuracy chart."""
    cs = results.get("shift", {}).get("cross_source", {})
    by = cs.get("by_source")
    if not by:
        return None
    srcs = sorted(by)
    vals = np.array([by[s]["length"]["reward_per_log_length_in_sd"] for s in srcs])
    y = np.arange(len(srcs))[::-1]

    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(srcs) + 1.5))
    ax.barh(y, vals, height=0.55, color=SERIES[0], zorder=3)
    ax.axvline(0, color=MUTED, lw=1.2)
    for yi, v in zip(y, vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), yi, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(srcs, fontsize=8.5)
    ax.set_xlabel("reward sd per unit log length")
    ax.set_title("Length coefficient by subset", loc="left")
    _clean(ax, xgrid=True)
    return _save(fig, out_dir, "fig_cross_source_length_bias.png")


def perturbation_effects(results: Dict, out_dir: str) -> Optional[str]:
    sp = results.get("shift", {}).get("surface_perturbation", {})
    by = sp.get("by_perturbation")
    if not by:
        return None
    names = sorted(by, key=lambda n: by[n]["reward_shift_in_sd"])
    vals = np.array([by[n]["reward_shift_in_sd"] for n in names])
    lenchg = [by[n]["changes_length"] for n in names]
    sd = results.get("shift", {}).get("reward_sd_reference", 1.0) or 1.0
    lo = np.array([by[n]["reward_shift_one_sided"]["ci_lo"] for n in names]) / sd
    hi = np.array([by[n]["reward_shift_one_sided"]["ci_hi"] for n in names]) / sd
    y = np.arange(len(names))[::-1]

    fig, ax = plt.subplots(figsize=(6.8, 0.44 * len(names) + 1.6))
    ax.hlines(y, lo, hi, color=INK_2, lw=1.2, zorder=3)
    for yi, v, ch in zip(y, vals, lenchg):
        ax.scatter([v], [yi], s=46, color=SERIES[1] if ch else SERIES[0], zorder=4,
                   edgecolor=SURFACE, linewidth=1.2)
    ax.axvline(0, color=MUTED, lw=1.2)
    span = float(np.nanmax(hi) - np.nanmin(lo)) or 1.0
    ax.set_xlim(np.nanmin(lo) - 0.05 * span, np.nanmax(hi) + 0.08 * span)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=8.5)
    ax.set_xlabel("reward shift (reward sd)")
    ax.set_title("What a surface rewrite is worth", loc="left")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=SERIES[0], label="length preserved"),
                       Line2D([], [], marker="o", ls="", color=SERIES[1], label="adds tokens")],
              loc="lower right", fontsize=8)
    _clean(ax, xgrid=True)
    return _save(fig, out_dir, "fig_perturbation_effects.png")


def make_all(results: Dict, out_dir: str, scored=None, unit: str = "tokens") -> Dict[str, str]:
    made = {}
    lp = results.get("length", {})
    for fn, args in (
        (length_decomposition, (lp, out_dir, unit)),
        (accuracy_by_length_gap, (lp, out_dir, unit)),
        (probe_effects, (results, out_dir)),
        (prefix_recovery, (results, out_dir)),
        (constraint_depth, (results, out_dir)),
        (cross_source_accuracy, (results, out_dir)),
        (cross_source_length_bias, (results, out_dir)),
        (perturbation_effects, (results, out_dir)),
    ):
        try:
            p = fn(*args)
            if p:
                made[fn.__name__] = p
        except Exception as e:  # a missing probe must not sink the report
            made[fn.__name__] = f"ERROR: {e}"
    if scored is not None:
        try:
            p = reward_vs_length(scored, out_dir, unit)
            if p:
                made["reward_vs_length"] = p
        except Exception as e:
            made["reward_vs_length"] = f"ERROR: {e}"
    return made
