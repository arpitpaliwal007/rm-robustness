"""Bradley-Terry reward model training.

loss = -log sigmoid( r(context, chosen) - r(context, rejected) )

Two options here are not standard boilerplate and exist for the study:

* `length_balanced`: resample the training pairs so that the sign of the length
  difference is uncorrelated with the preference label. An RM trained this way is the
  control arm for the headline question -- if accuracy collapses to chance, the model
  had nothing but length; if it holds, there is real signal underneath.
* `reward_l2`: BT loss is invariant to adding a constant to every reward, so scores
  drift during training and cross-run comparisons of raw reward become meaningless.
  A small L2 on the reward values anchors the scale. Off by default; report which.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import Pair
from .model import RewardModelBase


@dataclass
class TrainConfig:
    lr: float = 1e-5
    head_lr: Optional[float] = None  # defaults to lr; a larger head lr helps small backbones
    weight_decay: float = 0.01
    batch_size: int = 8           # pairs per step (2x sequences)
    grad_accum: int = 2
    epochs: float = 1.0
    max_steps: Optional[int] = None
    warmup_frac: float = 0.06
    max_grad_norm: float = 1.0
    reward_l2: float = 0.0
    seed: int = 0
    eval_every: int = 200
    log_every: int = 25
    amp: bool = True
    num_workers: int = 0
    length_balanced: bool = False
    length_unit: str = "chars"


class PairDataset(Dataset):
    def __init__(self, pairs: Sequence[Pair]):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Tuple[str, str, str]:
        p = self.pairs[i]
        return p.context_text(), p.chosen, p.rejected


def _collate(batch):
    ctx = [b[0] for b in batch]
    ch = [b[1] for b in batch]
    rj = [b[2] for b in batch]
    return ctx, ch, rj


def length_balanced_subset(pairs: Sequence[Pair], unit: str = "chars", seed: int = 0) -> List[Pair]:
    """Match the number of chosen-longer and chosen-shorter pairs, and match their
    |delta length| distributions by decile. The result is a training set in which
    'pick the longer one' is exactly a coin flip."""
    rng = random.Random(seed)

    def L(s: str) -> int:
        return len(s) if unit == "chars" else len(s.split())

    deltas = [L(p.chosen) - L(p.rejected) for p in pairs]
    pos = [(abs(d), i) for i, d in enumerate(deltas) if d > 0]
    neg = [(abs(d), i) for i, d in enumerate(deltas) if d < 0]
    if not pos or not neg:
        return list(pairs)
    mags = np.array([m for m, _ in pos + neg], dtype=float)
    edges = np.percentile(mags, np.linspace(0, 100, 11))
    edges[0], edges[-1] = -np.inf, np.inf

    keep: List[int] = []
    for b in range(10):
        lo, hi = edges[b], edges[b + 1]
        pb = [i for m, i in pos if lo <= m < hi]
        nb = [i for m, i in neg if lo <= m < hi]
        k = min(len(pb), len(nb))
        if k == 0:
            continue
        keep.extend(rng.sample(pb, k))
        keep.extend(rng.sample(nb, k))
    rng.shuffle(keep)
    return [pairs[i] for i in keep]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _param_groups(model: RewardModelBase, cfg: TrainConfig):
    head_lr = cfg.head_lr if cfg.head_lr is not None else cfg.lr
    head_names, body_names = [], []
    for n, _ in model.named_parameters():
        (head_names if ("head" in n or "classifier" in n or "score" in n or "pooler" in n) else body_names).append(n)
    named = dict(model.named_parameters())
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")

    def group(names, lr):
        decay = [named[n] for n in names if not any(nd in n for nd in no_decay)]
        nodecay = [named[n] for n in names if any(nd in n for nd in no_decay)]
        gs = []
        if decay:
            gs.append({"params": decay, "lr": lr, "weight_decay": cfg.weight_decay})
        if nodecay:
            gs.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})
        return gs

    return group(body_names, cfg.lr) + group(head_names, head_lr)


@torch.no_grad()
def evaluate(model: RewardModelBase, pairs: Sequence[Pair], batch_size: int = 16) -> Dict[str, float]:
    model.eval()
    ctxs = [p.context_text() for p in pairs]
    margins = []
    device = next(model.parameters()).device
    for i in range(0, len(pairs), batch_size):
        c = ctxs[i : i + batch_size]
        ch = [p.chosen for p in pairs[i : i + batch_size]]
        rj = [p.rejected for p in pairs[i : i + batch_size]]
        enc = model.encode(list(c) + list(c), list(ch) + list(rj))
        s = model.forward(enc.input_ids.to(device), enc.attention_mask.to(device)).float().cpu().numpy()
        n = len(c)
        margins.extend((s[:n] - s[n:]).tolist())
    m = np.asarray(margins)
    return {
        "acc": float((m > 0).mean()),
        "loss": float(np.mean(np.logaddexp(0.0, -m))),
        "mean_margin": float(m.mean()),
        "n": int(len(m)),
    }


def train(
    model: RewardModelBase,
    train_pairs: Sequence[Pair],
    val_pairs: Sequence[Pair],
    cfg: TrainConfig,
    out_dir: str,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pairs = list(train_pairs)
    balance_report = None
    if cfg.length_balanced:
        before = len(pairs)
        pairs = length_balanced_subset(pairs, unit=cfg.length_unit, seed=cfg.seed)
        balance_report = {"before": before, "after": len(pairs)}

    ds = PairDataset(pairs)
    dl = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate,
        num_workers=cfg.num_workers, drop_last=True,
    )
    steps_per_epoch = max(1, len(dl) // cfg.grad_accum)
    total_steps = cfg.max_steps or int(steps_per_epoch * cfg.epochs)
    warmup = max(1, int(cfg.warmup_frac * total_steps))

    opt = torch.optim.AdamW(_param_groups(model, cfg))
    base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    use_amp = cfg.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = os.path.join(out_dir, "train_log.jsonl")
    logf = open(log_path, "a")
    history: List[dict] = []
    best = {"acc": -1.0, "step": -1}
    step = 0
    t0 = time.time()
    micro = 0
    done = False
    model.train()

    while not done:
        for ctx, ch, rj in dl:
            enc = model.encode(list(ctx) + list(ctx), list(ch) + list(rj))
            ids = enc.input_ids.to(device)
            am = enc.attention_mask.to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=use_amp):
                s = model.forward(ids, am)
                n = len(ctx)
                r_c, r_r = s[:n], s[n:]
                loss = -F.logsigmoid(r_c - r_r).mean()
                if cfg.reward_l2 > 0:
                    loss = loss + cfg.reward_l2 * (r_c.pow(2).mean() + r_r.pow(2).mean())
            scaler.scale(loss / cfg.grad_accum).backward()
            micro += 1

            if micro % cfg.grad_accum == 0:
                scale = lr_at(step)
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * scale
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1

                if verbose and step % cfg.log_every == 0:
                    acc = float(((r_c - r_r) > 0).float().mean().detach())
                    rec = {"step": step, "loss": float(loss.detach()), "batch_acc": acc,
                           "lr": base_lrs[0] * scale, "elapsed": time.time() - t0}
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
                    print(f"step {step}/{total_steps} loss {float(loss.detach()):.4f} batch_acc {acc:.3f} "
                          f"({(time.time()-t0)/max(step,1):.2f}s/step)", flush=True)

                if step % cfg.eval_every == 0 or step == total_steps:
                    ev = evaluate(model, val_pairs)
                    ev.update({"step": step, "split": "val"})
                    history.append(ev)
                    logf.write(json.dumps(ev) + "\n")
                    logf.flush()
                    if verbose:
                        print(f"  [val] step {step} acc {ev['acc']:.4f} loss {ev['loss']:.4f}", flush=True)
                    if ev["acc"] > best["acc"]:
                        best = {"acc": ev["acc"], "step": step}
                        model.save(os.path.join(out_dir, "best"))
                    model.train()

                if step >= total_steps:
                    done = True
                    break
        if len(dl) == 0:
            break

    model.save(os.path.join(out_dir, "final"))
    logf.close()
    summary = {
        "config": asdict(cfg),
        "device": device,
        "n_train_pairs": len(pairs),
        "n_val_pairs": len(val_pairs),
        "total_steps": total_steps,
        "best_val": best,
        "val_history": history,
        "length_balance": balance_report,
        "wall_seconds": time.time() - t0,
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
