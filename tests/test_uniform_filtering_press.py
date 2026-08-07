# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for UniformFilteringPress — online per-token keep/skip decisions via majority vote.
"""

from dataclasses import dataclass

import pytest
import torch
from transformers import DynamicCache, pipeline

from kvpress import (
    KeyDiffPress,
    KnormPress,
    PrefillDecodingPress,
    StreamingLLMPress,
    TOVAPress,
    UniformFilteringPress,
)
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class FixedScorePress(ScorerPress):
    fixed_scores: torch.Tensor = None

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return self.fixed_scores


@pytest.fixture(scope="module")
def pipe():
    return pipeline("kv-press-text-generation", model="MaxJeblick/llama2-0b-unit-test", device_map="auto")


CONTEXT = "The quick brown fox jumps over the lazy dog. " * 10
QUESTION = "What animal jumps over the dog?"


def test_uniform_filtering_press_reduces_cache(pipe):
    """UniformFilteringPress should produce a smaller cache than no compression."""
    model = pipe.model
    tokenizer = pipe.tokenizer
    device = model.device

    input_ids = tokenizer.encode(CONTEXT, return_tensors="pt").to(device)

    cache_baseline = DynamicCache()
    with torch.no_grad():
        model.generate(input_ids, past_key_values=cache_baseline, max_new_tokens=20, do_sample=False)
    baseline_len = cache_baseline.get_seq_length()

    press = UniformFilteringPress(base_press=KnormPress(), target_compression_ratio=0.9)
    cache_filtered = DynamicCache()
    with torch.no_grad(), press(model):
        model.generate(input_ids, past_key_values=cache_filtered, max_new_tokens=20, do_sample=False)
    filtered_len = cache_filtered.get_seq_length()

    assert filtered_len < baseline_len, (
        f"filtered cache ({filtered_len}) should be smaller than baseline ({baseline_len})"
    )


def test_uniform_filtering_press_no_op_at_zero_ratio(pipe):
    """target_compression_ratio=0 should not filter any tokens."""
    cache_baseline = DynamicCache()
    pipe(CONTEXT, question=QUESTION, cache=cache_baseline, max_new_tokens=20)

    press = UniformFilteringPress(base_press=KnormPress(), target_compression_ratio=0.0)
    cache_filtered = DynamicCache()
    pipe(CONTEXT, question=QUESTION, press=press, cache=cache_filtered, max_new_tokens=20)

    for layer_idx in range(len(cache_baseline.layers)):
        assert cache_baseline.layers[layer_idx].keys.shape[2] == cache_filtered.layers[layer_idx].keys.shape[2]


def test_uniform_filtering_press_with_prefill_decoding(pipe):
    """UniformFilteringPress should work as decoding_press inside PrefillDecodingPress."""
    combined_press = PrefillDecodingPress(
        prefilling_press=KeyDiffPress(compression_ratio=0.5),
        decoding_press=UniformFilteringPress(base_press=KeyDiffPress(), target_compression_ratio=0.5),
    )

    cache = DynamicCache()
    result = pipe(CONTEXT, question=QUESTION, press=combined_press, cache=cache, max_new_tokens=15)

    assert len(result["answer"]) > 0, "No answer generated"


@pytest.mark.parametrize("scorer_cls", [KnormPress, KeyDiffPress, TOVAPress, StreamingLLMPress])
def test_uniform_filtering_press_with_different_scorers(pipe, scorer_cls):
    """UniformFilteringPress should work with any ScorerPress."""
    press = UniformFilteringPress(base_press=scorer_cls(), target_compression_ratio=0.5)

    cache = DynamicCache()
    result = pipe(CONTEXT, question=QUESTION, press=press, cache=cache, max_new_tokens=15)

    assert len(result["answer"]) > 0, f"No answer generated with {scorer_cls.__name__}"


def test_uniform_filtering_press_higher_ratio_filters_more(pipe):
    """Higher compression ratio should produce a smaller cache."""
    model = pipe.model
    tokenizer = pipe.tokenizer
    device = model.device

    input_ids = tokenizer.encode(CONTEXT, return_tensors="pt").to(device)

    cache_low = DynamicCache()
    press_low = UniformFilteringPress(base_press=KnormPress(), target_compression_ratio=0.3)
    with torch.no_grad(), press_low(model):
        model.generate(input_ids, past_key_values=cache_low, max_new_tokens=20, do_sample=False)

    cache_high = DynamicCache()
    press_high = UniformFilteringPress(base_press=KnormPress(), target_compression_ratio=0.7)
    with torch.no_grad(), press_high(model):
        model.generate(input_ids, past_key_values=cache_high, max_new_tokens=20, do_sample=False)

    low_len = cache_low.get_seq_length()
    high_len = cache_high.get_seq_length()
    assert high_len <= low_len, (
        f"higher ratio cache ({high_len}) should be <= lower ratio cache ({low_len})"
    )


def test_uniform_filtering_press_reuse_across_sequences(pipe):
    """Reusing a UniformFilteringPress across sequences should not crash."""
    press = UniformFilteringPress(base_press=KnormPress(), target_compression_ratio=0.5)

    model = pipe.model
    device = model.device
    long_ids = torch.arange(1, 81, dtype=torch.long, device=device).unsqueeze(0)
    short_ids = torch.arange(1, 9, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad(), press(model):
        model.generate(long_ids, max_new_tokens=6, do_sample=False)
        model.generate(short_ids, max_new_tokens=6, do_sample=False)


# --- Unit tests ---

BATCH, N_HEADS, SEQ_LEN, HEAD_DIM = 1, 2, 10, 4


def _make_press(scores, ratio=0.5):
    scorer = FixedScorePress()
    scorer.fixed_scores = scores
    return UniformFilteringPress(base_press=scorer, target_compression_ratio=ratio)


def _make_dummy_tensors(seq_len=SEQ_LEN):
    keys = torch.randn(BATCH, N_HEADS, seq_len, HEAD_DIM)
    values = torch.randn(BATCH, N_HEADS, seq_len, HEAD_DIM)
    hidden_states = torch.randn(BATCH, seq_len, HEAD_DIM)
    kwargs = {}
    return keys, values, hidden_states, kwargs


def _base_scores():
    """Scores where positions 0-4 are high (5.0) and 5-8 are low (1.0), last token varies."""
    scores = torch.zeros(BATCH, N_HEADS, SEQ_LEN)
    scores[:, :, :5] = 5.0
    scores[:, :, 5:9] = 1.0
    return scores


def test_compress_all_heads_accept():
    """Token kept when all heads accept."""
    scores = _base_scores()
    scores[:, :, -1] = 5.0
    press = _make_press(scores)
    keys, values, hidden_states, kwargs = _make_dummy_tensors()

    out_keys, out_values = press.compress(None, hidden_states, keys, values, None, kwargs)

    assert out_keys.shape[2] == SEQ_LEN


def test_compress_all_heads_reject():
    """Cache shrinks when all heads reject the new token."""
    scores = _base_scores()
    scores[:, :, -1] = 0.0
    press = _make_press(scores)
    keys, values, hidden_states, kwargs = _make_dummy_tensors()

    out_keys, out_values = press.compress(None, hidden_states, keys, values, None, kwargs)

    assert out_keys.shape[2] == SEQ_LEN - 1


def test_compress_tie_keeps_token():
    """With 2 heads, 1 accepting and 1 rejecting gives mean=0.5 >= 0.5 — token is kept."""
    scores = _base_scores()
    scores[:, 0, -1] = 5.0  # head 0 accepts
    scores[:, 1, -1] = 0.0  # head 1 rejects
    press = _make_press(scores)
    keys, values, hidden_states, kwargs = _make_dummy_tensors()

    out_keys, out_values = press.compress(None, hidden_states, keys, values, None, kwargs)

    assert out_keys.shape[2] == SEQ_LEN, "tie should keep the token"


def test_compress_no_op_when_n_kept_ge_k_len():
    """No filtering when n_kept >= k_len."""
    scores = _base_scores()
    scores[:, :, -1] = 0.0  # would normally be rejected
    press = _make_press(scores, ratio=0.0)
    keys, values, hidden_states, kwargs = _make_dummy_tensors()

    out_keys, out_values = press.compress(None, hidden_states, keys, values, None, kwargs)

    assert out_keys.shape[2] == SEQ_LEN, "should be no-op when n_kept >= k_len"
