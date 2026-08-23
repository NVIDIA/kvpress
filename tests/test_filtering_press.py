# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for FilteringPress — online per-token keep/skip decisions during decoding.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, pipeline

from kvpress import FilteringPress, KeyDiffPress, KnormPress, PrefillDecodingPress, StreamingLLMPress, TOVAPress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class KeyScorePress(ScorerPress):
    """Scorer that reads scores from keys[:, :, :, 0]. Embed desired scores there."""

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return keys[:, :, :, 0]


@pytest.fixture(scope="module")
def pipe():
    return pipeline("kv-press-text-generation", model="MaxJeblick/llama2-0b-unit-test", device_map="auto")


CONTEXT = "The quick brown fox jumps over the lazy dog. " * 10
QUESTION = "What animal jumps over the dog?"


def test_filtering_press_reduces_cache(pipe):
    """FilteringPress should produce a smaller cache than no compression."""
    model = pipe.model
    tokenizer = pipe.tokenizer
    device = model.device

    input_ids = tokenizer.encode(CONTEXT, return_tensors="pt").to(device)

    cache_baseline = DynamicCache()
    with torch.no_grad():
        model.generate(input_ids, past_key_values=cache_baseline, max_new_tokens=20, do_sample=False)
    baseline_len = cache_baseline.get_seq_length()

    press = FilteringPress(base_press=KnormPress(), target_compression_ratio=0.9)
    cache_filtered = DynamicCache()
    with torch.no_grad(), press(model):
        model.generate(input_ids, past_key_values=cache_filtered, max_new_tokens=20, do_sample=False)
    filtered_len = cache_filtered.get_seq_length()

    assert (
        filtered_len < baseline_len
    ), f"filtered cache ({filtered_len}) should be smaller than baseline ({baseline_len})"


def test_filtering_press_no_op_at_zero_ratio(pipe):
    """target_compression_ratio=0 should not filter any tokens."""
    cache_baseline = DynamicCache()
    pipe(CONTEXT, question=QUESTION, cache=cache_baseline, max_new_tokens=20)

    press = FilteringPress(base_press=KnormPress(), target_compression_ratio=0.0)
    cache_filtered = DynamicCache()
    pipe(CONTEXT, question=QUESTION, press=press, cache=cache_filtered, max_new_tokens=20)

    for layer_idx in range(len(cache_baseline.layers)):
        assert cache_baseline.layers[layer_idx].keys.shape[2] == cache_filtered.layers[layer_idx].keys.shape[2]


def test_filtering_press_with_prefill_decoding(pipe):
    """FilteringPress should work as decoding_press inside PrefillDecodingPress."""
    combined_press = PrefillDecodingPress(
        prefilling_press=KeyDiffPress(compression_ratio=0.5),
        decoding_press=FilteringPress(base_press=KeyDiffPress(), target_compression_ratio=0.5),
    )

    cache = DynamicCache()
    result = pipe(CONTEXT, question=QUESTION, press=combined_press, cache=cache, max_new_tokens=15)

    assert len(result["answer"]) > 0, "No answer generated"


@pytest.mark.parametrize("scorer_cls", [KnormPress, KeyDiffPress, TOVAPress, StreamingLLMPress])
def test_filtering_press_with_different_scorers(pipe, scorer_cls):
    """FilteringPress should work with any ScorerPress."""
    press = FilteringPress(base_press=scorer_cls(), target_compression_ratio=0.5)

    cache = DynamicCache()
    result = pipe(CONTEXT, question=QUESTION, press=press, cache=cache, max_new_tokens=15)

    assert len(result["answer"]) > 0, f"No answer generated with {scorer_cls.__name__}"


def test_filtering_press_higher_ratio_filters_more(pipe):
    """Higher compression ratio should produce a smaller cache."""
    model = pipe.model
    tokenizer = pipe.tokenizer
    device = model.device

    input_ids = tokenizer.encode(CONTEXT, return_tensors="pt").to(device)

    cache_low = DynamicCache()
    press_low = FilteringPress(base_press=KnormPress(), target_compression_ratio=0.3)
    with torch.no_grad(), press_low(model):
        model.generate(input_ids, past_key_values=cache_low, max_new_tokens=20, do_sample=False)

    cache_high = DynamicCache()
    press_high = FilteringPress(base_press=KnormPress(), target_compression_ratio=0.7)
    with torch.no_grad(), press_high(model):
        model.generate(input_ids, past_key_values=cache_high, max_new_tokens=20, do_sample=False)

    low_len = cache_low.get_seq_length()
    high_len = cache_high.get_seq_length()
    assert high_len <= low_len, f"higher ratio cache ({high_len}) should be <= lower ratio cache ({low_len})"


def test_filtering_press_reuse_across_sequences(pipe):
    """Reusing a FilteringPress across sequences should not crash."""
    press = FilteringPress(base_press=KnormPress(), target_compression_ratio=0.5)

    model = pipe.model
    device = model.device
    long_ids = torch.arange(1, 81, dtype=torch.long, device=device).unsqueeze(0)
    short_ids = torch.arange(1, 9, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad(), press(model):
        model.generate(long_ids, max_new_tokens=6, do_sample=False)
        model.generate(short_ids, max_new_tokens=6, do_sample=False)


# --- Unit tests ---

BATCH, N_HEADS, SEQ_LEN, HEAD_DIM = 1, 2, 10, 4


def _make_press(ratio=0.5):
    return FilteringPress(base_press=KeyScorePress(), target_compression_ratio=ratio)


def _make_module(layer_idx=0):
    return SimpleNamespace(layer_idx=layer_idx)


def _base_scores():
    """Scores where positions 0-4 are high (5.0) and 5-8 are low (1.0), last token varies."""
    scores = torch.zeros(BATCH, N_HEADS, SEQ_LEN)
    scores[:, :, :5] = 5.0
    scores[:, :, 5:9] = 1.0
    return scores


def _make_scored_tensors(scores):
    """Create test tensors with scores embedded in keys[:, :, :, 0]."""
    seq_len = scores.shape[2]
    keys = torch.randn(BATCH, N_HEADS, seq_len, HEAD_DIM)
    values = torch.randn(BATCH, N_HEADS, seq_len, HEAD_DIM)
    hidden_states = torch.randn(BATCH, seq_len, HEAD_DIM)
    kwargs = {"position_ids": torch.arange(seq_len).unsqueeze(0)}
    keys[:, :, :, 0] = scores
    return keys, values, hidden_states, kwargs


def test_compress_all_heads_accept():
    """Token kept at last position when all heads accept."""
    scores = _base_scores()
    scores[:, :, -1] = 5.0
    press = _make_press()
    module = _make_module()
    keys, values, hidden_states, kwargs = _make_scored_tensors(scores)

    out_keys, out_values = press.compress(module, hidden_states, keys, values, None, kwargs)

    assert out_keys.shape[2] == SEQ_LEN


def test_compress_all_heads_reject():
    """Cache shrinks when all heads reject the new token."""
    scores = _base_scores()
    scores[:, :, -1] = 0.0
    press = _make_press()
    module = _make_module()
    keys, values, hidden_states, kwargs = _make_scored_tensors(scores)

    out_keys, out_values = press.compress(module, hidden_states, keys, values, None, kwargs)

    assert out_keys.shape[2] == SEQ_LEN - 1


def test_compress_one_head_rejects():
    """Shape unchanged when one head rejects; rejected position is added to masked_key_indices."""
    scores = _base_scores()
    scores[:, 0, -1] = 5.0
    scores[:, 1, -1] = 0.0
    press = _make_press()
    module = _make_module()
    keys, values, hidden_states, kwargs = _make_scored_tensors(scores)

    out_keys, out_values = press.compress(module, hidden_states, keys, values, None, kwargs)

    assert out_keys.shape[2] == SEQ_LEN
    assert hasattr(module, "masked_key_indices")
    b_idx, h_idx, s_idx = module.masked_key_indices
    assert (h_idx == 1).all(), "only head 1 should be masked"
    assert (s_idx == SEQ_LEN - 1).all(), "masked position should be the last token"


def test_compress_masked_indices_accumulate():
    """Masked indices accumulate across multiple compress calls."""
    scores1 = _base_scores()
    scores1[:, 0, -1] = 5.0
    scores1[:, 1, -1] = 0.0
    press = _make_press()
    module = _make_module()

    keys1, values1, hidden_states1, kwargs1 = _make_scored_tensors(scores1)
    press.compress(module, hidden_states1, keys1, values1, None, kwargs1)

    scores2 = torch.zeros(BATCH, N_HEADS, 11)
    scores2[:, :, :5] = 5.0
    scores2[:, :, 5:10] = 1.0
    scores2[:, 0, -1] = 0.0
    scores2[:, 1, -1] = 5.0
    keys2, values2, hidden_states2, kwargs2 = _make_scored_tensors(scores2)

    press.compress(module, hidden_states2, keys2, values2, None, kwargs2)

    b_idx, h_idx, s_idx = module.masked_key_indices
    assert len(h_idx) == 2, "should have 2 masked positions total"


def test_compress_filters_even_with_small_cache():
    """Filtering applies even when the cache is smaller than n_kept."""
    small_seq = 3
    scores = torch.tensor([[[5.0, 5.0, 0.0], [5.0, 5.0, 0.0]]])
    press = _make_press(ratio=0.5)
    module = _make_module()
    keys, values, hidden_states, kwargs = _make_scored_tensors(scores)

    out_keys, out_values = press.compress(module, hidden_states, keys, values, None, kwargs)

    assert out_keys.shape[2] == small_seq - 1
