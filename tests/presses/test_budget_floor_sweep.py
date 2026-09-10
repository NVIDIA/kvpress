# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the budget floor sweep deferred by #264.

Three budget derivations still floor to zero on short contexts, silently
breaking partial compression:

* ``KVComposePress`` computed ``int(numel * (1 - compression_ratio))``, so a
  ratio close to 1.0 yields ``topk(0)`` and empties the whole cache (or a
  single layer's cache after the global budget distribution) although the
  caller only compresses partially.
* ``CriticalKVPress`` computed ``int((1 - ratio) * k_len * first_stage_ratio)``
  which is 0 on short contexts, so ``topk(scores, 0)`` silently skips the
  stage-1 selection.
* ``FastKVzipPress`` computed ``int(ctx_len * window_ratio)`` which is 0 on
  short contexts, and assigning to ``scores[:, :, -0:]`` protects the entire
  context instead of a local window.

The tests below drive the real presses on the unit-test model (no mocks of the
code under test) with degenerate ratios close to 1.0 on short contexts.
"""

import pytest
import torch
from transformers import DynamicCache

from kvpress import CriticalKVPress, FastKVzipPress, KnormPress, KVComposePress
from kvpress.presses.fastkvzip_press import FastKVzipGate
from tests.fixtures import unit_test_model  # noqa: F401


class MockGatesFastKVzipPress(FastKVzipPress):
    """Test version of FastKVzipPress that creates a mock gate instead of loading from HuggingFace."""

    def post_init_from_model(self, model):
        if self.gates is None:
            dtype = model.config.dtype
            input_dim = model.config.hidden_size
            ngroup = model.config.num_attention_heads // model.config.num_key_value_heads
            nhead = model.config.num_key_value_heads

            self.gates = []
            for idx in range(model.config.num_hidden_layers):
                module = FastKVzipGate(idx, input_dim, nhead, ngroup, dtype).to(model.device)
                self.gates.append(module)


class StoreScoresCriticalKVPress(CriticalKVPress):
    """Records the final scores returned by the real CriticalKVPress.score during compression."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_scores = []

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        scores = super().score(module, hidden_states, keys, values, attentions, kwargs)
        self.stored_scores.append(scores)
        return scores


def compress_short_context(press, model, context_len=8, batch_size=1):
    """Run a single prefill through ``press`` on the unit-test model and return the cache."""
    with press(model):
        input_ids = torch.randint(0, 1024, (batch_size, context_len), device=model.device)
        cache = DynamicCache()
        model(input_ids, past_key_values=cache)
    return cache


@pytest.mark.parametrize("compression_ratio", [0.95, 0.99])
def test_kvcompose_structured_never_empties_a_layer_on_short_contexts(unit_test_model, compression_ratio):  # noqa: F811
    """``int(numel * (1 - compression_ratio))`` floors to 0 on short contexts and ``topk(0)``
    silently empties every layer cache although the caller only compresses partially."""
    press = KVComposePress(compression_ratio=compression_ratio)
    cache = compress_short_context(press, unit_test_model)

    assert cache.get_seq_length() >= 1
    for layer in cache.layers:
        assert layer.keys.shape[2] >= 1


@pytest.mark.parametrize("compression_ratio", [0.95, 0.99])
def test_kvcompose_keep_token_lower_bound_survives_on_short_contexts(unit_test_model, compression_ratio):  # noqa: F811
    """The +1e9 score boost cannot rescue the lower-bound tokens when the global budget floors
    to 0 (``topk(0)`` drops boosted tokens too), so every layer must keep at least
    ``keep_token_lower_bound`` tokens."""
    press = KVComposePress(compression_ratio=compression_ratio, keep_token_lower_bound=2)
    cache = compress_short_context(press, unit_test_model)

    assert cache.get_seq_length() >= press.keep_token_lower_bound
    for layer in cache.layers:
        assert layer.keys.shape[2] >= press.keep_token_lower_bound


def test_kvcompose_structured_batch_size_2_does_not_collapse_on_short_contexts(unit_test_model):  # noqa: F811
    """The same zero budget also collapses a batch of sequences to an empty cache."""
    press = KVComposePress(compression_ratio=0.95)
    cache = compress_short_context(press, unit_test_model, batch_size=2)

    assert cache.get_seq_length() >= 1
    for layer in cache.layers:
        assert layer.keys.shape[2] >= 1


def test_kvcompose_unstructured_keeps_at_least_one_token_on_short_contexts(unit_test_model):  # noqa: F811
    """Unstructured compression flags evicted tokens via ``masked_key_indices``; with a zero
    budget every token of every head is flagged for eviction. Per-head zero counts remain
    legitimate head pruning, but the global budget must still keep at least one token."""
    context_len = 8
    press = KVComposePress(structured=False, compression_ratio=0.99)
    compress_short_context(press, unit_test_model, context_len=context_len)

    config = unit_test_model.config
    n_masked = sum(len(layer.self_attn.masked_key_indices[2]) for layer in unit_test_model.model.layers)
    total = config.num_hidden_layers * config.num_key_value_heads * context_len
    assert n_masked < total


@pytest.mark.parametrize("compression_ratio", [0.95, 0.99])
def test_criticalkv_stage1_selection_boosts_on_short_contexts(unit_test_model, compression_ratio):  # noqa: F811
    """``int((1 - compression_ratio) * k_len * first_stage_ratio)`` floors to 0 on short contexts
    and ``topk(scores, 0)`` legally returns nothing, silently skipping stage-1 protection."""
    press = StoreScoresCriticalKVPress(press=KnormPress(compression_ratio=compression_ratio))
    compress_short_context(press, unit_test_model)

    assert len(press.stored_scores) == unit_test_model.config.num_hidden_layers
    for scores in press.stored_scores:
        boosted = scores == torch.finfo(scores.dtype).max
        assert boosted.any(dim=-1).all(), "stage-1 selection boosted no position"


def test_criticalkv_stage1_selection_disabled_when_first_stage_ratio_is_zero(unit_test_model):  # noqa: F811
    """``first_stage_ratio=0`` disables stage 1 by design; the floor must not re-enable it."""
    press = StoreScoresCriticalKVPress(press=KnormPress(compression_ratio=0.95), first_stage_ratio=0.0)
    compress_short_context(press, unit_test_model)

    for scores in press.stored_scores:
        assert not (scores == torch.finfo(scores.dtype).max).any()


def test_fastkvzip_short_context_window_does_not_protect_everything(unit_test_model):  # noqa: F811
    """``int(ctx_len * window_ratio)`` floors to 0 on short contexts, and assigning to
    ``scores[:, :, -0:]`` protects the entire context instead of a local window."""
    press = MockGatesFastKVzipPress(compression_ratio=0.5)
    compress_short_context(press, unit_test_model)

    scores = press.score_val  # stacked per-layer scores, shape (n_layers, bsz, heads, ctx_len)
    assert not (scores == 1.0).all()
    assert (scores[..., -1] == 1.0).all()
