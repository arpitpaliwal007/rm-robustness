"""Surface features of a response. Kept deliberately cheap and tokenizer-agnostic.

Length is measured three ways because they disagree and the disagreement matters:
characters (what a human eyeballs), whitespace words (what most papers report), and
model tokens (what the RM actually sees, and what a truncation limit acts on).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

_WORD_RE = re.compile(r"\S+")
_SENT_RE = re.compile(r"[.!?]+(?:\s|$)")
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")

HEDGES = (
    "i think", "i believe", "maybe", "perhaps", "probably", "it seems",
    "i'm not sure", "i am not sure", "might", "possibly", "as far as i know",
)
AGREEMENT = (
    "you're right", "you are right", "you're correct", "you are correct",
    "good point", "i agree", "absolutely", "exactly", "great question",
    "that's true", "that is true", "yes,",
)
REFUSAL = (
    "i can't", "i cannot", "i won't", "i will not", "i'm not able",
    "i am not able", "i'm sorry", "i am sorry", "i don't think i should",
    "that's not something", "i shouldn't",
)


@dataclass
class Features:
    n_chars: int
    n_words: int
    n_tokens: int
    n_sentences: int
    n_bullets: int
    mean_word_len: float
    type_token_ratio: float
    n_question_marks: int
    has_hedge: int
    has_agreement: int
    has_refusal: int
    uppercase_frac: float

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def featurize(text: str, token_counter: Optional[Callable[[str], int]] = None) -> Features:
    words = _WORD_RE.findall(text)
    lower = text.lower()
    n_words = len(words)
    n_tokens = token_counter(text) if token_counter is not None else n_words
    uniq = len({w.lower() for w in words})
    return Features(
        n_chars=len(text),
        n_words=n_words,
        n_tokens=int(n_tokens),
        n_sentences=max(1, len(_SENT_RE.findall(text))),
        n_bullets=len(_BULLET_RE.findall(text)),
        mean_word_len=(sum(len(w) for w in words) / n_words) if n_words else 0.0,
        type_token_ratio=(uniq / n_words) if n_words else 0.0,
        n_question_marks=text.count("?"),
        has_hedge=int(any(h in lower for h in HEDGES)),
        has_agreement=int(any(a in lower for a in AGREEMENT)),
        has_refusal=int(any(r in lower for r in REFUSAL)),
        uppercase_frac=(sum(c.isupper() for c in text) / len(text)) if text else 0.0,
    )


def log_len(n: float, eps: float = 1.0) -> float:
    """Log length with a floor. Delta of this is the standard length covariate:
    it makes the 10-vs-20-token gap and the 200-vs-400-token gap comparable, which
    a raw difference does not."""
    return math.log(max(float(n), eps))


LENGTH_UNITS = ("chars", "words", "tokens")


def length_of(f: Features, unit: str) -> int:
    return {"chars": f.n_chars, "words": f.n_words, "tokens": f.n_tokens}[unit]
