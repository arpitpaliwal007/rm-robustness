"""A reward model whose behaviour is written by the test, not learned.

Every probe is a measuring instrument. The way to know an instrument works is to point
it at something whose value you already know. These fixtures inject a known length bias,
a known sycophancy effect, a known position sensitivity -- and the tests assert that the
probe reads back the number that was put in.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from rmrobust.model import RewardModelBase


class ScriptedRewardModel(RewardModelBase):
    """Scores (context, response) with an arbitrary python function."""

    def __init__(self, fn: Callable[[str, str], float], max_length: int = 512, noise: float = 0.0, seed: int = 0):
        super().__init__()
        self.fn = fn
        self.max_length = max_length
        self.noise = noise
        self._rng = np.random.default_rng(seed)
        self._p = nn.Parameter(torch.zeros(1), requires_grad=False)  # gives .parameters() a device

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def score(self, contexts, responses, batch_size: int = 16, return_meta: bool = False, progress: bool = False):
        vals = np.array([self.fn(c, r) for c, r in zip(contexts, responses)], dtype=float)
        if self.noise:
            vals = vals + self._rng.normal(0.0, self.noise, size=len(vals))
        if not return_meta:
            return vals
        meta = [{"n_response_tokens": len(r.split()), "response_truncated": False,
                 "context_truncated": False} for r in responses]
        return vals, meta


def length_only(alpha: float = 1.0) -> Callable[[str, str], float]:
    return lambda c, r: alpha * math.log(max(len(r.split()), 1))


def oracle(chosen_texts: Sequence[str], bonus: float = 1.0) -> Callable[[str, str], float]:
    keep = set(chosen_texts)
    return lambda c, r: (bonus if r in keep else 0.0)


def bag_of_words() -> Callable[[str, str], float]:
    """Order-blind by construction: depends only on the multiset of non-space characters,
    so any reordering of the same text scores identically."""
    return lambda c, r: sum(ord(ch) % 7 for ch in r if not ch.isspace()) / 100.0


def first_k_words(k: int = 12) -> Callable[[str, str], float]:
    """Reads only the opening of the response."""
    return lambda c, r: sum(ord(ch) % 7 for ch in " ".join(r.split()[:k])) / 100.0


def recent_context_only(window_chars: int = 240) -> Callable[[str, str], float]:
    """Sees only the tail of the context, so anything said earlier is invisible."""

    def fn(c: str, r: str) -> float:
        tail = c[-window_chars:]
        return 0.001 * len(set(tail.lower().split()) & set(r.lower().split()))

    return fn


def constraint_aware(window_chars: int = 260, bonus: float = 1.0) -> Callable[[str, str], float]:
    """Rewards the compliant response, but only when the constraint that defines
    compliance is still inside the model's context window.

    This is the exact behaviour the constraint-depth probe is built to detect, written
    out explicitly: a model with a hard attention horizon. The probe should read a full
    gap at depth 0 and a vanishing one once the constraint slides past the horizon.
    """
    from rmrobust.probes import position as P

    doc = P.load_probeset()
    items = [i for i in doc["items"] if i["arm"] == "constraint_depth"]
    compliant = {i["response_compliant"]: i["constraint"] for i in items}

    def fn(c: str, r: str) -> float:
        con = compliant.get(r)
        if con is None:
            return 0.0
        return bonus if con in c[-window_chars:] else 0.0

    return fn
