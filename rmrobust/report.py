"""Turn the results JSON into a readable markdown report.

The report states each number next to the thing that would make it uninteresting: the
accuracy next to the length baseline, the sycophancy effect next to its placebo floor,
the position effect next to the order-blindness control, the OOD accuracy next to the
in-domain one. A reward model probe result read on its own is almost always
over-interpreted, so the format does the pairing rather than trusting the reader to.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional


def _fmt(e: Optional[Dict], digits: int = 3, signed: bool = False) -> str:
    if not isinstance(e, dict) or "value" not in e:
        return "n/a"
    v, lo, hi = e.get("value"), e.get("ci_lo"), e.get("ci_hi")
    s = f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"
    if lo is None or hi is None or lo != lo:
        return s
    return f"{s} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _sec(title: str) -> str:
    return f"\n## {title}\n\n"


def render(results: Dict, unit: str = "tokens") -> str:
    meta = results.get("meta", {})
    out: List[str] = []
    out.append(f"# Reward model robustness report\n")
    if meta:
        out.append(
            f"Model `{meta.get('model_name','?')}` · trained on "
            f"`{', '.join(meta.get('train_sources', []))}` · "
            f"{meta.get('n_train_pairs','?')} pairs · seed {meta.get('seed','?')}\n"
        )
        out.append(f"Evaluated on {meta.get('n_eval_pairs','?')} held-out pairs "
                   f"from `{', '.join(meta.get('eval_sources', []))}`.\n")

    # ---------------------------------------------------------------- headline
    L = results.get("length", {}).get("overall", {}).get(unit, {})
    if L and not L.get("insufficient_data"):
        c, d, st = L["comparative"], L["decomposition"], L["stratified"]
        lef = d["length_explained_fraction"]
        out.append(_sec("Headline"))
        headline = (
            f"- Reward model accuracy: **{_fmt(c['acc_rm'])}**\n"
            f"- Length alone (calibrated, out-of-fold): **{_fmt(c['acc_length_logistic_oof'])}**\n"
            f"- All surface features, no semantics: **{_fmt(c['acc_surface_logistic_oof'])}**\n"
            f"- Accuracy with a monotone length model subtracted: **{_fmt(d['acc_residual_after_length'])}**\n"
        )
        if lef.get("reliable", True):
            headline += (f"- **Share of above-chance accuracy explained by length: "
                         f"{lef['value']:+.1%} [{lef['ci_lo']:+.1%}, {lef['ci_hi']:+.1%}]**\n")
        else:
            headline += (
                f"- Share of above-chance accuracy explained by length: **not estimable** "
                f"(point estimate {lef['value']:+.1%}). The accuracy confidence interval "
                f"includes chance, so the denominator of this ratio is indistinguishable "
                f"from zero. Train longer or evaluate on more pairs before quoting it.\n"
            )
        out.append(headline)
        out.append(
            "\nRead the last line as a fraction of *skill*, not of variance. A value near 1 "
            "means removing everything a monotone function of length could have supplied "
            "takes the model to chance. A negative value means length is a net drag: the "
            "model would be more accurate without it.\n"
        )
        out.append(
            f"\nAccuracy restricted to pairs where the two responses are within 5% of each "
            f"other in length: **{_fmt(st.get('acc_length_neutral_5pct'))}** "
            f"(n={st.get('acc_length_neutral_5pct',{}).get('n','?')}).\n"
        )
        cor = L["correlational"]
        out.append(
            f"\nReward is a strongly monotone function of length: isotonic R² = "
            f"{cor['r2_isotonic_reward_on_loglen']:.3f}, Spearman "
            f"{cor['spearman_reward_vs_len']:+.3f}, slope "
            f"{cor['reward_slope_per_100_units_in_sd']:+.3f} reward sd per 100 {unit}.\n"
        )
        trunc = results["length"].get("truncation", {})
        if isinstance(trunc.get("untruncated_only"), dict) and "decomposition" in trunc["untruncated_only"]:
            u = trunc["untruncated_only"]["decomposition"]["length_explained_fraction"]
            frac_t = trunc["frac_pairs_with_truncated_response"]
            if u.get("reliable", True):
                out.append(
                    f"\nWith truncated pairs excluded ({frac_t:.1%} of the set), the "
                    f"length-explained share is {u['value']:+.1%}.\n"
                )
            else:
                out.append(
                    f"\n{frac_t:.1%} of pairs had a response cut by the length limit. On the "
                    f"remainder the length-explained share is also not estimable at this "
                    f"accuracy.\n"
                )

    # ---------------------------------------------------------------- by source
    bs = results.get("length", {}).get("by_source", {})
    if bs:
        out.append(_sec("Length bias by subset"))
        out.append("| subset | n | RM acc | length-only acc | length-explained share | chosen longer |\n")
        out.append("|---|---:|---:|---:|---:|---:|\n")
        for src, r in sorted(bs.items()):
            if r.get("insufficient_data"):
                continue
            out.append(
                f"| `{src}` | {r['n']} | {r['comparative']['acc_rm']['value']:.3f} | "
                f"{r['comparative']['acc_length_logistic_oof']['value']:.3f} | "
                f"{r['decomposition']['length_explained_fraction']['value']:+.1%}"
                f"{'' if r['decomposition']['length_explained_fraction'].get('reliable', True) else ' *'} | "
                f"{r['frac_chosen_longer']:.1%} |\n"
            )
        out.append(
            "\n`*` marks a row whose accuracy CI includes chance, where the "
            "length-explained share is not estimable.\n"
            "\nThe `chosen longer` column is a property of HH, not of the model: the helpful "
            "subsets prefer the longer response and harmless-base prefers the shorter one. A "
            "single length coefficient cannot be right for both.\n"
        )

    # ---------------------------------------------------------------- sycophancy
    syc = results.get("sycophancy")
    if syc:
        out.append(_sec("Sycophancy"))
        sf, pl, net = syc.get("stance_flip", {}), syc.get("placebo", {}), syc.get("stance_minus_placebo", {})
        if sf.get("n_items"):
            out.append(
                f"- Echoing the user's stated stance is worth "
                f"**{_fmt(sf['sycophancy_in_reward_sd'], signed=True)} reward sd** "
                f"({sf['n_items']} items over {sf['n_topics']} topics, cluster-bootstrapped by topic)\n"
                f"- Placebo floor, same design with an irrelevant stance: "
                f"{_fmt(pl.get('sycophancy_in_reward_sd'), signed=True)} reward sd\n"
            )
            if net:
                out.append(f"- **Net of placebo: {net['in_reward_sd']:+.3f} reward sd** "
                           f"(raw {net['value']:+.4f} [{net['ci_lo']:+.4f}, {net['ci_hi']:+.4f}])\n")
            out.append(
                f"- The same response is preferred under the matching stance in "
                f"{sf['echo_win_rate_per_response']:.1%} of cases\n"
            )
            out.append(
                "\nThe response text is byte-identical across the two conditions, so length, "
                "style and content cancel exactly. The placebo arm is the instrument's floor; "
                "subtract it before believing the primary number.\n"
            )
        cap, fla = syc.get("capitulation", {}), syc.get("flattery", {})
        if cap.get("n_items"):
            la = cap["length_adjusted"]
            out.append(
                f"\n- Abandoning a correct answer under user pushback: "
                f"{_fmt(cap['delta_in_reward_sd'], signed=True)} reward sd, "
                f"win rate {cap['win_rate']:.1%}; at equal length "
                f"{la['intercept']:+.4f} [{la['intercept_ci'][0]:+.4f}, {la['intercept_ci'][1]:+.4f}]\n"
            )
        if fla.get("n_items"):
            la = fla["length_adjusted"]
            out.append(
                f"- Praise over critique of the user's own work: "
                f"{_fmt(fla['delta_in_reward_sd'], signed=True)} reward sd, "
                f"win rate {fla['win_rate']:.1%}; at equal length "
                f"{la['intercept']:+.4f} [{la['intercept_ci'][0]:+.4f}, {la['intercept_ci'][1]:+.4f}]\n"
            )
            out.append(
                "\nThese two arms compare different texts, so unlike the stance-flip arm they "
                "carry a length confound. Items are matched to within about 15%, and the "
                "'at equal length' figure is the intercept of the effect regressed on the "
                "length difference.\n"
            )

    # ---------------------------------------------------------------- position
    pos = results.get("position")
    if pos:
        out.append(_sec("Position"))
        inv = pos.get("segment_swap", {}).get("order_invariant_items", {})
        ac = pos.get("segment_swap", {}).get("answer_vs_caveat_items", {})
        rev = pos.get("sentence_reversal", {})
        if inv.get("n_items"):
            out.append(
                f"- Reordering two interchangeable segments (identical tokens) moves the reward by "
                f"**{inv['mean_abs_effect_in_reward_sd']:.3f} reward sd** on average; "
                f"signed effect {_fmt(inv.get('signed_effect_segment1_first_minus_segment2_first'), 4, signed=True)}\n"
            )
        if ac.get("n_items"):
            out.append(
                f"- Leading with the answer rather than the caveat: "
                f"{_fmt(ac.get('signed_effect_answer_first_minus_caveat_first'), 4, signed=True)} raw, "
                f"{ac['signed_effect_in_reward_sd']:+.3f} reward sd\n"
            )
        if rev.get("n_responses"):
            out.append(
                f"- Control: reversing sentence order in real HH responses shifts reward by "
                f"{rev['mean_abs_shift_in_sd']:.3f} sd, original preferred "
                f"{rev['frac_original_preferred']:.1%} of the time. "
                f"{'The model registers order at all.' if rev['mean_abs_shift_in_sd'] > 0.05 else 'The model is close to order-blind, which makes the swap result above uninformative.'}\n"
            )
        cd = pos.get("constraint_depth", {})
        if cd.get("by_depth"):
            ks = sorted(cd["by_depth"], key=int)
            first, last = cd["by_depth"][ks[0]], cd["by_depth"][ks[-1]]
            out.append(
                f"\n- A constraint stated immediately before the question produces a "
                f"compliant-minus-violating gap of {_fmt(first['compliance_gap'], 4)}; "
                f"pushed back by {ks[-1]} filler turn pairs it is {_fmt(last['compliance_gap'], 4)} "
                f"(retention {cd.get('retention_ratio_deep_over_shallow', float('nan')):.2f})\n"
            )
        pt = pos.get("prefix_truncation", {})
        if pt.get("by_prefix_length"):
            out.append(f"\n| response tokens visible | accuracy | share of full skill | agrees with full |\n")
            out.append("|---:|---:|---:|---:|\n")
            for r in pt["by_prefix_length"]:
                out.append(
                    f"| {r['k_tokens']} | {r['acc_from_prefix']['value']:.3f} | "
                    f"{r['frac_of_full_skill_recovered']:.1%} | {r['decision_agreement_with_full']:.1%} |\n"
                )
            out.append(f"| full | {pt['acc_full_response']:.3f} | 100.0% | 100.0% |\n")
            out.append(
                "\nShare of full skill is (acc_k - 0.5) / (acc_full - 0.5), so it can exceed "
                "100% or go negative when the model is near chance; read it alongside the "
                "accuracy column rather than on its own.\n"
            )

    # ---------------------------------------------------------------- shift
    sh = results.get("shift")
    if sh:
        out.append(_sec("Distribution shift"))
        cs = sh.get("cross_source", {})
        by = cs.get("by_source", {})
        if by:
            out.append(f"Reference (in-distribution) subset: `{cs.get('reference_source')}`.\n\n")
            out.append("| subset | n | acc | drop vs ref | reward mean shift (ref sd) | KS vs ref | length coef (sd/log len) | ECE |\n")
            out.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for src, r in sorted(by.items()):
                out.append(
                    f"| `{src}` | {r['n_pairs']} | {r['acc']['value']:.3f} | "
                    f"{r['acc_drop_vs_reference']:+.3f} | "
                    f"{r['reward_mean_shift_vs_reference_in_ref_sd']:+.2f} | "
                    f"{r['ks_vs_reference']:.3f} | "
                    f"{r['length']['reward_per_log_length_in_sd']:+.3f} | "
                    f"{r['calibration']['ece']:.3f} |\n"
                )
            out.append(
                "\nThe reward-mean-shift column is the one that matters for RLHF: a reward "
                "whose scale moves between domains breaks any optimiser that assumed a fixed "
                "scale, and it moves without the accuracy column necessarily moving at all.\n"
            )
        pert = sh.get("surface_perturbation", {}).get("by_perturbation", {})
        if pert:
            out.append("\n| rewrite | adds tokens | reward shift (sd) | reward up | decision flips |\n")
            out.append("|---|:--:|---:|---:|---:|\n")
            for name, r in sorted(pert.items(), key=lambda kv: -kv[1]["reward_shift_in_sd"]):
                out.append(
                    f"| {name.replace('_',' ')} | {'yes' if r['changes_length'] else 'no'} | "
                    f"{r['reward_shift_in_sd']:+.3f} | {r['frac_reward_increased']:.1%} | "
                    f"{r['decision_flip_rate_both_sided']:.1%} |\n"
                )
            out.append(
                "\n`decision flips` applies the rewrite to *both* responses, so a robust model "
                "should score 0%. `reward shift` applies it to one, and is what a policy would "
                "collect for adopting the habit.\n"
            )

    figs = results.get("figures", {})
    if figs:
        out.append(_sec("Figures"))
        for k, v in figs.items():
            if isinstance(v, str) and not v.startswith("ERROR"):
                out.append(f"- `{v}`\n")

    out.append(_sec("How to read a null"))
    out.append(
        "A probe that returns zero can mean the model is robust, or it can mean the probe "
        "did not move anything the model reads. Each arm here ships with the control that "
        "distinguishes those: the placebo arm for sycophancy, sentence reversal for position, "
        "the length-neutral subset for length, and the in-domain row for shift. Report the "
        "control next to the effect or the null is not interpretable.\n"
    )
    return "".join(out)


def write(results: Dict, path: str, unit: str = "tokens") -> str:
    md = render(results, unit=unit)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return md
