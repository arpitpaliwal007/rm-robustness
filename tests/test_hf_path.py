"""Exercise the Hugging Face code path without a model hub.

The T4 run uses `HFRewardModel`, which downloads weights. This environment has no hub
access, so the test builds the same class around a locally-constructed BERT and a
locally-trained tokenizer. Everything except the download is the real path: the
truncation policy, the special-token handling, the batch collation, the Bradley-Terry
step, checkpoint save and reload, and scoring.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from rmrobust import data as D
from rmrobust.model import HFRewardModel, load_model
from rmrobust.probes import length as P_length
from rmrobust.scoring import score_pairs
from rmrobust.train import TrainConfig, evaluate, train

DATA_DIR = "data"


def _stub_tokenizer(vocab_size: int = 2000):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    pairs = D.load_pairs(DATA_DIR, "helpful-base", "test", limit=300)
    corpus = [p.context_text() + " " + p.chosen for p in pairs] + [p.rejected for p in pairs]
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(
        corpus,
        trainers.WordPieceTrainer(vocab_size=vocab_size,
                                  special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]),
    )
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]")


@pytest.fixture(scope="module")
def hf_model():
    from transformers import BertConfig, BertForSequenceClassification

    tokenizer = _stub_tokenizer()
    cfg = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=64, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=128, max_position_embeddings=256,
                     num_labels=1, pad_token_id=tokenizer.pad_token_id)
    m = HFRewardModel.__new__(HFRewardModel)
    nn.Module.__init__(m)
    m.model_name = "local-test-bert"
    m.max_length = 128
    m.min_context_tokens = 16
    m.tokenizer = tokenizer
    m.backbone = BertForSequenceClassification(cfg)
    m._prefix_ids, m._suffix_ids = m._special_affixes()
    return m


def test_special_affixes_detected(hf_model):
    assert len(hf_model._prefix_ids) == 1 and len(hf_model._suffix_ids) == 1


def test_encoding_respects_the_length_budget_and_keeps_the_response(hf_model):
    pairs = D.load_pairs(DATA_DIR, "helpful-online", "test", limit=40)  # the long subset
    enc = hf_model.encode([p.context_text() for p in pairs], [p.chosen for p in pairs])
    assert enc.input_ids.shape[1] <= hf_model.max_length
    assert enc.attention_mask.shape == enc.input_ids.shape
    # padding is masked out
    pad = hf_model.tokenizer.pad_token_id
    assert bool(((enc.input_ids == pad) & (enc.attention_mask == 1)).sum() == 0)
    # some context was dropped rather than the response, on at least one long example
    assert any(enc.context_truncated)
    for i, p in enumerate(pairs):
        if not enc.response_truncated[i]:
            assert enc.n_response_tokens[i] == hf_model.count_tokens(p.chosen)


def test_forward_and_score_shapes(hf_model):
    pairs = D.load_pairs(DATA_DIR, "helpful-base", "test", limit=24)
    s, meta = hf_model.score([p.context_text() for p in pairs], [p.chosen for p in pairs],
                             batch_size=8, return_meta=True, progress=False)
    assert s.shape == (24,)
    assert np.isfinite(s).all()
    assert len(meta) == 24


def test_scoring_is_batch_invariant(hf_model):
    """A response's reward must not depend on what it was batched with."""
    pairs = D.load_pairs(DATA_DIR, "helpful-base", "test", limit=16)
    ctx = [p.context_text() for p in pairs]
    res = [p.chosen for p in pairs]
    a = hf_model.score(ctx, res, batch_size=16, progress=False)
    b = hf_model.score(ctx, res, batch_size=1, progress=False)
    assert np.allclose(a, b, atol=1e-4), np.abs(a - b).max()


def test_train_step_reduces_the_bradley_terry_loss(hf_model, tmp_path):
    pairs = D.load_pairs(DATA_DIR, "helpful-base", "train", limit=200)
    tr, va = D.train_val_split(pairs, 0.2, seed=0)
    before = evaluate(hf_model, va, batch_size=8)
    cfg = TrainConfig(lr=3e-4, batch_size=4, grad_accum=1, max_steps=30, eval_every=30,
                      log_every=100, amp=False, seed=0)
    summary = train(hf_model, tr, va, cfg, out_dir=str(tmp_path), device="cpu", verbose=False)
    after = summary["val_history"][-1]
    assert after["loss"] < before["loss"], (before, after)
    assert os.path.exists(os.path.join(str(tmp_path), "final", "rm_config.json"))


def test_checkpoint_roundtrip(hf_model, tmp_path):
    path = str(tmp_path / "ckpt")
    hf_model.save(path)
    reloaded = load_model(path, device="cpu")
    pairs = D.load_pairs(DATA_DIR, "helpful-base", "test", limit=12)
    ctx = [p.context_text() for p in pairs]
    res = [p.chosen for p in pairs]
    a = hf_model.score(ctx, res, batch_size=6, progress=False)
    b = reloaded.score(ctx, res, batch_size=6, progress=False)
    assert np.allclose(a, b, atol=1e-4)
    assert reloaded.max_length == hf_model.max_length


def test_probe_runs_on_the_hf_path(hf_model):
    pairs = D.load_pairs(DATA_DIR, "helpful-base", "test", limit=120)
    sp = score_pairs(hf_model, pairs, batch_size=8, progress=False)
    r = P_length.run(sp, units=("tokens",), per_source=False, n_boot=100)
    assert "decomposition" in r["overall"]["tokens"]
    assert 0.0 <= r["overall"]["tokens"]["comparative"]["acc_rm"]["value"] <= 1.0
