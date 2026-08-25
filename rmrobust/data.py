"""HH-RLHF loading and dialogue parsing.

The raw dataset gives two full transcripts per example, `chosen` and `rejected`,
in the format

    \n\nHuman: <text>\n\nAssistant: <text>\n\nHuman: ... \n\nAssistant: <text>

The two transcripts share a prefix and diverge at some assistant turn. Every probe
in this study needs the *response* isolated from the *context*, so parsing is not a
detail: if you compare full transcripts you are comparing shared context tokens too,
and every length statistic is contaminated by prompt length.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Iterator, List, Optional, Sequence, Tuple

SUBSETS = ("helpful-base", "helpful-online", "helpful-rejection-sampled", "harmless-base")

# GitHub serves helpful-* through the LFS media endpoint and harmless-base straight
# from the object store; both are mirrored here so the repo works without HF access.
_GH_LFS = "https://media.githubusercontent.com/media/anthropics/hh-rlhf/master/{sub}/{split}.jsonl.gz"
_GH_RAW = "https://raw.githubusercontent.com/anthropics/hh-rlhf/master/{sub}/{split}.jsonl.gz"

_TURN_RE = re.compile(r"\n\n(Human|Assistant):[ ]?")


@dataclass
class Turn:
    role: str  # "Human" | "Assistant"
    text: str


@dataclass
class Pair:
    """One preference pair with the divergent continuation isolated from context."""

    uid: str
    source: str
    split: str
    context_turns: List[Turn]
    chosen: str
    rejected: str
    # bookkeeping used by the probes
    n_context_turns: int = 0
    diverged_early: bool = False  # divergence is not at the final assistant turn
    degenerate: bool = False  # empty or identical responses

    def context_text(self, add_generation_prefix: bool = True) -> str:
        return render_turns(self.context_turns, add_generation_prefix=add_generation_prefix)

    def to_json(self) -> dict:
        d = asdict(self)
        d["context_turns"] = [(t.role, t.text) for t in self.context_turns]
        return d


def render_turns(turns: Sequence[Turn], add_generation_prefix: bool = True) -> str:
    """Render turns back into HH transcript format.

    `add_generation_prefix` appends the trailing "\n\nAssistant:" that the response
    continues from, which is what the reward model should condition on.
    """
    parts = [f"\n\n{t.role}: {t.text}" for t in turns]
    out = "".join(parts)
    if add_generation_prefix:
        out += "\n\nAssistant:"
    return out


def parse_transcript(text: str) -> Optional[List[Turn]]:
    """Split an HH transcript into turns. Returns None if it does not parse."""
    if not text:
        return None
    matches = list(_TURN_RE.finditer(text))
    if not matches:
        return None
    # Anything before the first role marker is junk; HH transcripts start with "\n\nHuman:".
    turns: List[Turn] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        turns.append(Turn(role=m.group(1), text=text[start:end].strip()))
    return turns


def _divergence_index(a: Sequence[Turn], b: Sequence[Turn]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i].role != b[i].role or a[i].text != b[i].text:
            return i
    return n


def parse_pair(record: dict, uid: str, source: str, split: str) -> Optional[Pair]:
    ct = parse_transcript(record.get("chosen", ""))
    rt = parse_transcript(record.get("rejected", ""))
    if not ct or not rt:
        return None

    i = _divergence_index(ct, rt)
    if i >= len(ct) or i >= len(rt):
        # One transcript is a strict prefix of the other: the longer one continues
        # past where the shorter stops. Treat the shared part as context and the
        # remainder as the response (the shorter side then has an empty response).
        i = min(len(ct), len(rt), max(i - 1, 0))

    context = list(ct[:i])
    chosen = render_turns(ct[i:], add_generation_prefix=False).lstrip("\n")
    rejected = render_turns(rt[i:], add_generation_prefix=False).lstrip("\n")
    # Strip the leading role marker from the divergent turn; it is supplied by the
    # generation prefix on the context side.
    chosen = re.sub(r"^(Human|Assistant):[ ]?", "", chosen)
    rejected = re.sub(r"^(Human|Assistant):[ ]?", "", rejected)

    diverged_early = not (i == len(ct) - 1 and i == len(rt) - 1)
    degenerate = (not chosen.strip()) or (not rejected.strip()) or (chosen.strip() == rejected.strip())

    return Pair(
        uid=uid,
        source=source,
        split=split,
        context_turns=context,
        chosen=chosen,
        rejected=rejected,
        n_context_turns=len(context),
        diverged_early=diverged_early,
        degenerate=degenerate,
    )


def _raw_path(data_dir: str, subset: str, split: str) -> str:
    return os.path.join(data_dir, "raw", f"{subset}_{split}.jsonl.gz")


def iter_raw(data_dir: str, subset: str, split: str) -> Iterator[dict]:
    path = _raw_path(data_dir, subset, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/fetch_data.py --data-dir {data_dir}` first."
        )
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_pairs(
    data_dir: str,
    subset: str,
    split: str,
    limit: Optional[int] = None,
    drop_degenerate: bool = True,
    drop_early_divergence: bool = False,
) -> List[Pair]:
    """Load and parse one (subset, split).

    `drop_early_divergence=True` keeps only pairs whose divergence is the final
    assistant turn. Those are the clean single-response comparisons; the rest branch
    earlier and contain extra human turns inside the "response". Default False, with
    the flag recorded on each pair so probes can stratify.
    """
    out: List[Pair] = []
    for i, rec in enumerate(iter_raw(data_dir, subset, split)):
        p = parse_pair(rec, uid=f"{subset}/{split}/{i}", source=subset, split=split)
        if p is None:
            continue
        if drop_degenerate and p.degenerate:
            continue
        if drop_early_divergence and p.diverged_early:
            continue
        out.append(p)
        if limit is not None and len(out) >= limit:
            break
    return out


def load_many(
    data_dir: str,
    subsets: Sequence[str],
    split: str,
    limit_per_subset: Optional[int] = None,
    **kw,
) -> List[Pair]:
    pairs: List[Pair] = []
    for s in subsets:
        pairs.extend(load_pairs(data_dir, s, split, limit=limit_per_subset, **kw))
    return pairs


def train_val_split(pairs: Sequence[Pair], val_frac: float = 0.05, seed: int = 0) -> Tuple[List[Pair], List[Pair]]:
    import random

    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_val = int(round(val_frac * len(pairs)))
    val = [pairs[i] for i in idx[:n_val]]
    train = [pairs[i] for i in idx[n_val:]]
    return train, val


def parse_stats(pairs: Sequence[Pair]) -> dict:
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "frac_early_divergence": sum(p.diverged_early for p in pairs) / n,
        "mean_context_turns": sum(p.n_context_turns for p in pairs) / n,
        "mean_chosen_chars": sum(len(p.chosen) for p in pairs) / n,
        "mean_rejected_chars": sum(len(p.rejected) for p in pairs) / n,
    }
