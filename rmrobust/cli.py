"""Command line entry points.

    python -m rmrobust.cli train  --backbone microsoft/deberta-v3-small --out runs/base
    python -m rmrobust.cli probe  --checkpoint runs/base/best --out runs/base
    python -m rmrobust.cli report --results runs/base/results.json
    python -m rmrobust.cli all    --out runs/base            # train, probe, report

Everything is seeded and every run writes its full config next to its results, because
the numbers here are small enough that an unrecorded flag change looks like a finding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from . import data as D
from . import figures as F
from . import report as R
from .model import build_model, load_model
from .probes import length as P_length
from .probes import position as P_position
from .probes import shift as P_shift
from .probes import sycophancy as P_sycophancy
from .scoring import score_pairs
from .train import TrainConfig, train


def _add_common(p):
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=16)


def _add_model(p):
    p.add_argument("--backbone", default="distilroberta-base",
                   help="HF model id, or 'tiny' for the offline test backbone")
    p.add_argument("--max-length", type=int, default=512)


def _train_sources(a) -> List[str]:
    return a.train_sources.split(",") if a.train_sources else ["helpful-base"]


def _eval_sources(a) -> List[str]:
    return a.eval_sources.split(",") if a.eval_sources else list(D.SUBSETS)


def cmd_train(a) -> Dict:
    sources = _train_sources(a)
    pairs = D.load_many(a.data_dir, sources, "train", limit_per_subset=a.limit_train)
    tr, va = D.train_val_split(pairs, val_frac=a.val_frac, seed=a.seed)
    print(f"train pairs {len(tr)}  val pairs {len(va)}  sources {sources}", flush=True)

    kind = "tiny" if a.backbone == "tiny" else "hf"
    kw = {"max_length": a.max_length}
    if kind == "hf":
        kw.update({"model_name": a.backbone, "gradient_checkpointing": a.grad_checkpointing})
    model = build_model(kind, **kw)

    cfg = TrainConfig(
        lr=a.lr, head_lr=a.head_lr, batch_size=a.batch_size, grad_accum=a.grad_accum,
        epochs=a.epochs, max_steps=a.max_steps, seed=a.seed, eval_every=a.eval_every,
        reward_l2=a.reward_l2, length_balanced=a.length_balanced, amp=not a.no_amp,
    )
    summary = train(model, tr, va, cfg, out_dir=a.out, device=a.device)
    summary["train_sources"] = sources
    summary["backbone"] = a.backbone
    with open(os.path.join(a.out, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _train_meta(out_dir: str) -> Dict:
    """Carry the training config into the results file. A probe result whose training
    run cannot be identified is not reproducible."""
    path = os.path.join(out_dir, "train_summary.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            ts = json.load(f)
    except Exception:
        return {}
    return {
        "n_train_pairs": ts.get("n_train_pairs"),
        "train_steps": ts.get("total_steps"),
        "best_val_acc": (ts.get("best_val") or {}).get("acc"),
        "length_balanced": (ts.get("config") or {}).get("length_balanced"),
        "backbone": ts.get("backbone"),
    }


def cmd_probe(a) -> Dict:
    t0 = time.time()
    ckpt = a.checkpoint or os.path.join(a.out, "best")
    model = load_model(ckpt, device=a.device or ("cuda" if _cuda() else "cpu"))
    eval_sources = _eval_sources(a)
    ref = a.reference_source or _train_sources(a)[0]

    pairs_by_source = {
        s: D.load_pairs(a.data_dir, s, "test", limit=a.limit_eval) for s in eval_sources
    }
    all_pairs = [p for s in eval_sources for p in pairs_by_source[s]]
    ref_pairs = pairs_by_source.get(ref) or all_pairs
    print(f"scoring {len(all_pairs)} eval pairs across {len(eval_sources)} subsets", flush=True)

    sp_all = score_pairs(model, all_pairs, batch_size=a.batch_size, progress=not a.quiet)
    reward_sd = float(np.concatenate([sp_all.r_chosen, sp_all.r_rejected]).std())
    print(f"reward sd on eval set: {reward_sd:.4f}", flush=True)

    results: Dict = {
        "meta": {
            "checkpoint": ckpt,
            "model_name": getattr(model, "model_name", "tiny"),
            "train_sources": _train_sources(a),
            "eval_sources": eval_sources,
            "reference_source": ref,
            "n_eval_pairs": len(all_pairs),
            "seed": a.seed,
            "reward_sd": reward_sd,
            "max_length": getattr(model, "max_length", None),
            **_train_meta(a.out),
        },
        "length": P_length.run(sp_all, seed=a.seed, n_boot=a.n_boot),
    }
    if not a.skip_sycophancy:
        print("probe: sycophancy", flush=True)
        results["sycophancy"] = P_sycophancy.run(
            model, reward_sd=reward_sd, batch_size=a.batch_size, n_boot=a.n_boot,
            seed=a.seed, progress=False)
    if not a.skip_position:
        print("probe: position", flush=True)
        results["position"] = P_position.run(
            model, pairs=ref_pairs, reward_sd=reward_sd, batch_size=a.batch_size,
            n_boot=a.n_boot, seed=a.seed, max_hh_pairs=a.max_position_pairs, progress=False)
    if not a.skip_shift:
        print("probe: distribution shift", flush=True)
        results["shift"] = P_shift.run(
            model, pairs_by_source, reference_source=ref, reward_sd=reward_sd,
            batch_size=a.batch_size, n_boot=a.n_boot, seed=a.seed,
            max_perturbation_pairs=a.max_perturbation_pairs, progress=False)
        results["shift"]["reward_sd_reference"] = reward_sd

    results["figures"] = F.make_all(results, os.path.join(a.out, "figures"), scored=sp_all)
    results["meta"]["probe_wall_seconds"] = time.time() - t0

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    R.write(results, os.path.join(a.out, "report.md"))
    print(f"wrote {a.out}/results.json and {a.out}/report.md in {time.time()-t0:.0f}s", flush=True)
    return results


def cmd_report(a) -> str:
    with open(a.results) as f:
        results = json.load(f)
    out = a.out_file or os.path.join(os.path.dirname(a.results), "report.md")
    md = R.write(results, out)
    print(md)
    return md


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("rmrobust")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    _add_common(t); _add_model(t)
    t.add_argument("--out", default="runs/base")
    t.add_argument("--train-sources", default="helpful-base")
    t.add_argument("--limit-train", type=int, default=None)
    t.add_argument("--val-frac", type=float, default=0.05)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--head-lr", type=float, default=None)
    t.add_argument("--grad-accum", type=int, default=2)
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--eval-every", type=int, default=200)
    t.add_argument("--reward-l2", type=float, default=0.0)
    t.add_argument("--length-balanced", action="store_true",
                   help="control arm: resample so length carries no preference signal")
    t.add_argument("--grad-checkpointing", action="store_true")
    t.add_argument("--no-amp", action="store_true")
    t.set_defaults(fn=cmd_train)

    p = sub.add_parser("probe")
    _add_common(p)
    p.add_argument("--out", default="runs/base")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--train-sources", default="helpful-base")
    p.add_argument("--eval-sources", default=None)
    p.add_argument("--reference-source", default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--max-position-pairs", type=int, default=500)
    p.add_argument("--max-perturbation-pairs", type=int, default=400)
    p.add_argument("--skip-sycophancy", action="store_true")
    p.add_argument("--skip-position", action="store_true")
    p.add_argument("--skip-shift", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_probe)

    r = sub.add_parser("report")
    r.add_argument("--results", required=True)
    r.add_argument("--out-file", default=None)
    r.set_defaults(fn=cmd_report)

    a = sub.add_parser("all")
    _add_common(a); _add_model(a)
    a.add_argument("--out", default="runs/base")
    a.add_argument("--train-sources", default="helpful-base")
    a.add_argument("--eval-sources", default=None)
    a.add_argument("--reference-source", default=None)
    a.add_argument("--limit-train", type=int, default=None)
    a.add_argument("--limit-eval", type=int, default=None)
    a.add_argument("--val-frac", type=float, default=0.05)
    a.add_argument("--lr", type=float, default=1e-5)
    a.add_argument("--head-lr", type=float, default=None)
    a.add_argument("--grad-accum", type=int, default=2)
    a.add_argument("--epochs", type=float, default=1.0)
    a.add_argument("--max-steps", type=int, default=None)
    a.add_argument("--eval-every", type=int, default=200)
    a.add_argument("--reward-l2", type=float, default=0.0)
    a.add_argument("--length-balanced", action="store_true")
    a.add_argument("--grad-checkpointing", action="store_true")
    a.add_argument("--no-amp", action="store_true")
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--max-position-pairs", type=int, default=500)
    a.add_argument("--max-perturbation-pairs", type=int, default=400)
    a.add_argument("--skip-sycophancy", action="store_true")
    a.add_argument("--skip-position", action="store_true")
    a.add_argument("--skip-shift", action="store_true")
    a.add_argument("--quiet", action="store_true")
    a.set_defaults(fn=lambda ns: (cmd_train(ns), setattr(ns, "checkpoint", os.path.join(ns.out, "best")), cmd_probe(ns))[-1])

    return ap


def main(argv: Optional[List[str]] = None):
    ap = build_parser()
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    main()
