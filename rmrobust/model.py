"""Reward model: a scalar head over a transformer encoder.

Two backends behind one interface.

* `HFRewardModel` is the one you train on a T4. Any AutoModelForSequenceClassification
  backbone with `num_labels=1` works; deberta-v3-small and distilroberta-base both fit
  comfortably in 16GB at seq 512.
* `TinyRewardModel` is a self-contained ~1M-param transformer with a hash tokenizer and
  no network dependency. It exists so the whole pipeline -- training, every probe, the
  report -- can be run and tested in an environment with no GPU and no model hub access.
  It is a test fixture, not a scientific instrument; do not report numbers from it.

Truncation policy is a real design decision, not plumbing. The response must survive.
Contexts are truncated from the LEFT (drop the oldest dialogue) and responses from the
RIGHT, and every scoring call reports whether truncation happened, because a probe that
compares a 600-token response against a 200-token one under a 512-token limit is
measuring the truncator, not the reward model.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncodeResult:
    input_ids: torch.Tensor       # (B, L)
    attention_mask: torch.Tensor  # (B, L)
    n_response_tokens: List[int]  # response tokens actually kept
    response_truncated: List[bool]
    context_truncated: List[bool]


class RewardModelBase(nn.Module):
    """Interface every probe depends on."""

    max_length: int = 512

    def count_tokens(self, text: str) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def truncate_to_tokens(self, text: str, k: int) -> str:
        """First k tokens of `text`, decoded back to a string.

        The base implementation cuts on whitespace, which is what a model without a
        reversible tokenizer has to do. Subclasses with a real tokenizer override it so
        that the prefix probe cuts where the model actually sees a boundary.
        """
        parts = text.split()
        return " ".join(parts[:k])

    def encode(self, contexts: Sequence[str], responses: Sequence[str]) -> EncodeResult:  # pragma: no cover
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    @torch.no_grad()
    def score(
        self,
        contexts: Sequence[str],
        responses: Sequence[str],
        batch_size: int = 16,
        return_meta: bool = False,
        progress: bool = False,
    ):
        """Scalar reward per (context, response). Never call this in a training loop."""
        self.eval()
        device = next(self.parameters()).device
        out: List[float] = []
        meta: List[dict] = []
        rng = range(0, len(contexts), batch_size)
        if progress:
            try:
                from tqdm.auto import tqdm

                rng = tqdm(list(rng), desc="scoring")
            except Exception:
                pass
        for i in rng:
            c = list(contexts[i : i + batch_size])
            r = list(responses[i : i + batch_size])
            enc = self.encode(c, r)
            ids = enc.input_ids.to(device)
            am = enc.attention_mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                s = self.forward(ids, am)
            out.extend(s.float().detach().cpu().numpy().tolist())
            if return_meta:
                for j in range(len(c)):
                    meta.append(
                        {
                            "n_response_tokens": enc.n_response_tokens[j],
                            "response_truncated": enc.response_truncated[j],
                            "context_truncated": enc.context_truncated[j],
                        }
                    )
        arr = np.asarray(out, dtype=float)
        return (arr, meta) if return_meta else arr


# --------------------------------------------------------------------------------------
# Hugging Face backbone
# --------------------------------------------------------------------------------------


class HFRewardModel(RewardModelBase):
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        max_length: int = 512,
        min_context_tokens: int = 32,
        dropout: Optional[float] = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.min_context_tokens = min_context_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            # GPT-2 family: no pad token. Reuse eos and tell the model about it.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        cfg = AutoConfig.from_pretrained(model_name, num_labels=1)
        cfg.pad_token_id = self.tokenizer.pad_token_id
        if dropout is not None:
            for k in ("hidden_dropout_prob", "attention_probs_dropout_prob", "dropout", "attn_pdrop", "resid_pdrop"):
                if hasattr(cfg, k):
                    setattr(cfg, k, dropout)
        self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        self._prefix_ids, self._suffix_ids = self._special_affixes()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_to_tokens(self, text: str, k: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:k]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _special_affixes(self):
        """Discover the special tokens this tokenizer wraps a single sequence in.

        Encode a sentinel with and without special tokens and diff the two. This works
        across tokenizer implementations and across transformers versions, where the
        older `build_inputs_with_special_tokens` hook is not always present -- and it
        matters here because those ids are what the truncation budget has to leave room
        for.
        """
        tok = self.tokenizer
        try:
            bare = tok.encode("robustness", add_special_tokens=False)
            full = tok.encode("robustness", add_special_tokens=True)
        except Exception:
            return [], []
        if not bare or len(full) < len(bare):
            return [], []
        for i in range(len(full) - len(bare) + 1):
            if full[i : i + len(bare)] == bare:
                return list(full[:i]), list(full[i + len(bare):])
        return [], []

    def encode(self, contexts: Sequence[str], responses: Sequence[str]) -> EncodeResult:
        tok = self.tokenizer
        pre, suf = self._prefix_ids, self._suffix_ids
        budget = self.max_length - len(pre) - len(suf)

        ids_batch, ctx_trunc, resp_trunc, n_resp = [], [], [], []
        for c, r in zip(contexts, responses):
            c_ids = tok.encode(c, add_special_tokens=False)
            r_ids = tok.encode(r, add_special_tokens=False)
            ct = rt = False
            if len(c_ids) + len(r_ids) > budget:
                # 1. shrink the context from the left, down to min_context_tokens
                keep_ctx = max(self.min_context_tokens, budget - len(r_ids))
                if keep_ctx < len(c_ids):
                    c_ids = c_ids[-keep_ctx:]
                    ct = True
                # 2. if still over, cut the response tail
                if len(c_ids) + len(r_ids) > budget:
                    r_ids = r_ids[: budget - len(c_ids)]
                    rt = True
            ids_batch.append(list(pre) + c_ids + r_ids + list(suf))
            ctx_trunc.append(ct)
            resp_trunc.append(rt)
            n_resp.append(len(r_ids))

        L = max(len(x) for x in ids_batch)
        pad = tok.pad_token_id
        input_ids = torch.full((len(ids_batch), L), pad, dtype=torch.long)
        attn = torch.zeros((len(ids_batch), L), dtype=torch.long)
        for i, x in enumerate(ids_batch):
            input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
            attn[i, : len(x)] = 1
        return EncodeResult(input_ids, attn, n_resp, resp_trunc, ctx_trunc)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits.squeeze(-1)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "rm_config.json"), "w") as f:
            json.dump(
                {
                    "kind": "hf",
                    "model_name": self.model_name,
                    "max_length": self.max_length,
                    "min_context_tokens": self.min_context_tokens,
                },
                f,
                indent=2,
            )


# --------------------------------------------------------------------------------------
# Offline tiny backbone (test fixture)
# --------------------------------------------------------------------------------------


class HashTokenizer:
    """Word-level hashing tokenizer. Deterministic across processes (no PYTHONHASHSEED
    dependence) because it uses an explicit FNV-1a rather than builtin hash()."""

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self._reserved = 3

    @staticmethod
    def _fnv1a(s: str) -> int:
        h = 0xCBF29CE484222325
        for ch in s.encode("utf-8"):
            h ^= ch
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return h

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        toks = text.lower().split()
        ids = [self._reserved + (self._fnv1a(t) % (self.vocab_size - self._reserved)) for t in toks]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        return ids


class TinyRewardModel(RewardModelBase):
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_length: int = 256,
        min_context_tokens: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = HashTokenizer(vocab_size)
        self.max_length = max_length
        self.min_context_tokens = min_context_tokens
        self.cfg = dict(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            max_length=max_length, min_context_tokens=min_context_tokens, dropout=dropout,
        )
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_length + 2, d_model)
        import warnings
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head.bias)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def encode(self, contexts: Sequence[str], responses: Sequence[str]) -> EncodeResult:
        tok = self.tokenizer
        budget = self.max_length - 2
        ids_batch, ctx_trunc, resp_trunc, n_resp = [], [], [], []
        for c, r in zip(contexts, responses):
            c_ids = tok.encode(c)
            r_ids = tok.encode(r)
            ct = rt = False
            if len(c_ids) + len(r_ids) > budget:
                keep_ctx = max(self.min_context_tokens, budget - len(r_ids))
                if keep_ctx < len(c_ids):
                    c_ids = c_ids[-keep_ctx:]
                    ct = True
                if len(c_ids) + len(r_ids) > budget:
                    r_ids = r_ids[: budget - len(c_ids)]
                    rt = True
            ids_batch.append([tok.cls_token_id] + c_ids + r_ids + [tok.sep_token_id])
            ctx_trunc.append(ct)
            resp_trunc.append(rt)
            n_resp.append(len(r_ids))
        L = max(len(x) for x in ids_batch)
        input_ids = torch.zeros((len(ids_batch), L), dtype=torch.long)
        attn = torch.zeros((len(ids_batch), L), dtype=torch.long)
        for i, x in enumerate(ids_batch):
            input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
            attn[i, : len(x)] = 1
        return EncodeResult(input_ids, attn, n_resp, resp_trunc, ctx_trunc)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).clamp(max=self.pos.num_embeddings - 1)
        h = self.emb(input_ids) + self.pos(pos)[None]
        h = self.enc(h, src_key_padding_mask=(attention_mask == 0))
        h = self.norm(h)
        # mean-pool over real tokens (the tiny model has no [CLS] pretraining to lean on)
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "tiny_rm.pt"))
        with open(os.path.join(path, "rm_config.json"), "w") as f:
            json.dump({"kind": "tiny", **self.cfg}, f, indent=2)


def build_model(kind: str = "hf", **kw) -> RewardModelBase:
    if kind == "hf":
        return HFRewardModel(**kw)
    if kind == "tiny":
        return TinyRewardModel(**kw)
    raise ValueError(f"unknown model kind {kind!r}")


def load_model(path: str, device: str = "cpu") -> RewardModelBase:
    with open(os.path.join(path, "rm_config.json")) as f:
        cfg = json.load(f)
    kind = cfg.pop("kind")
    if kind == "hf":
        m = HFRewardModel.__new__(HFRewardModel)
        nn.Module.__init__(m)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        m.model_name = cfg["model_name"]
        m.max_length = cfg["max_length"]
        m.min_context_tokens = cfg.get("min_context_tokens", 32)
        m.tokenizer = AutoTokenizer.from_pretrained(path)
        m.backbone = AutoModelForSequenceClassification.from_pretrained(path)
        m._prefix_ids, m._suffix_ids = m._special_affixes()
    else:
        m = TinyRewardModel(**cfg)
        m.load_state_dict(torch.load(os.path.join(path, "tiny_rm.pt"), map_location="cpu"))
    return m.to(device)
