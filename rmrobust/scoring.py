"""Score a set of preference pairs once and hand the same arrays to every probe.

Scoring is the expensive part of the study; the probes are arithmetic on top of it.
`ScoredPairs` also carries the token counts the model actually saw (post-truncation),
which is the length that matters -- not the length of the string you passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .data import Pair
from .features import Features, featurize
from .model import RewardModelBase


@dataclass
class ScoredPairs:
    pairs: List[Pair]
    r_chosen: np.ndarray
    r_rejected: np.ndarray
    feat_chosen: List[Features]
    feat_rejected: List[Features]
    meta_chosen: List[dict]
    meta_rejected: List[dict]

    @property
    def margin(self) -> np.ndarray:
        return self.r_chosen - self.r_rejected

    @property
    def correct(self) -> np.ndarray:
        m = self.margin
        c = (m > 0).astype(float)
        c[m == 0] = 0.5
        return c

    @property
    def sources(self) -> np.ndarray:
        return np.array([p.source for p in self.pairs])

    def lengths(self, unit: str = "tokens") -> tuple:
        if unit == "tokens":
            a = np.array([m["n_response_tokens"] for m in self.meta_chosen], dtype=float)
            b = np.array([m["n_response_tokens"] for m in self.meta_rejected], dtype=float)
            return a, b
        key = {"chars": "n_chars", "words": "n_words"}[unit]
        a = np.array([getattr(f, key) for f in self.feat_chosen], dtype=float)
        b = np.array([getattr(f, key) for f in self.feat_rejected], dtype=float)
        return a, b

    def truncated_mask(self) -> np.ndarray:
        """Pairs where either response lost tokens to the length limit. Every
        length claim should be checked with these excluded."""
        return np.array(
            [mc["response_truncated"] or mr["response_truncated"]
             for mc, mr in zip(self.meta_chosen, self.meta_rejected)]
        )

    def subset(self, mask: np.ndarray) -> "ScoredPairs":
        idx = np.where(np.asarray(mask))[0]
        return ScoredPairs(
            pairs=[self.pairs[i] for i in idx],
            r_chosen=self.r_chosen[idx],
            r_rejected=self.r_rejected[idx],
            feat_chosen=[self.feat_chosen[i] for i in idx],
            feat_rejected=[self.feat_rejected[i] for i in idx],
            meta_chosen=[self.meta_chosen[i] for i in idx],
            meta_rejected=[self.meta_rejected[i] for i in idx],
        )

    def __len__(self) -> int:
        return len(self.pairs)


def score_pairs(
    model: RewardModelBase,
    pairs: Sequence[Pair],
    batch_size: int = 16,
    progress: bool = True,
) -> ScoredPairs:
    pairs = list(pairs)
    ctx = [p.context_text() for p in pairs]
    ch = [p.chosen for p in pairs]
    rj = [p.rejected for p in pairs]
    tc = model.count_tokens
    r_c, meta_c = model.score(ctx, ch, batch_size=batch_size, return_meta=True, progress=progress)
    r_r, meta_r = model.score(ctx, rj, batch_size=batch_size, return_meta=True, progress=progress)
    return ScoredPairs(
        pairs=pairs,
        r_chosen=r_c,
        r_rejected=r_r,
        feat_chosen=[featurize(t, tc) for t in ch],
        feat_rejected=[featurize(t, tc) for t in rj],
        meta_chosen=meta_c,
        meta_rejected=meta_r,
    )
