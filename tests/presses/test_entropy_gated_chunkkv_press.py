# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import pytest
import torch
from torch import nn

from kvpress import EntropyGatedChunkKVPress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class FixedScorer(ScorerPress):
    """Scorer returning pre-set token scores, so chunk statistics are fully controlled."""

    scores: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return self.scores


class DummyAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim


def run_press(scores, press):
    """Run compress on keys whose values encode their position, and return the kept positions."""
    kv_len = scores.shape[2]
    n_heads, head_dim = scores.shape[1], 4
    positions = torch.arange(kv_len, dtype=torch.float32)
    keys = positions.view(1, 1, kv_len, 1).expand(1, n_heads, kv_len, head_dim).contiguous()
    values = keys.clone()
    out_keys, out_values = press.compress(DummyAttention(head_dim), None, keys, values, None, {})
    assert torch.equal(out_keys, out_values)
    return out_keys[0, 0, :, 0].long().tolist()


def expected_budget(kv_len, compression_ratio):
    """The token budget the press targets, matching ChunkKVPress."""
    return max(1, int(kv_len * (1 - compression_ratio)))


def spiky_scores(n_chunks, chunk_length, n_heads=2):
    """One dominant needle per chunk, the rest near-zero: every chunk is important and spiky."""
    kv_len = n_chunks * chunk_length
    scores = torch.full((1, n_heads, kv_len), 0.01)
    needles = [i * chunk_length + (i % chunk_length) for i in range(n_chunks)]
    for rank, idx in enumerate(needles):
        scores[0, :, idx] = 10.0 + rank  # distinct so chunk ordering is deterministic
    return scores, needles


@pytest.mark.parametrize("compression_ratio", [0.1, 0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("chunk_length", [1, 4, 10, 20])
@pytest.mark.parametrize("kv_len", [100, 251])
def test_retains_exact_budget(compression_ratio, chunk_length, kv_len):
    """The retained token count matches ChunkKVPress's budget exactly, including partial chunks."""
    torch.manual_seed(0)
    scores = torch.rand(1, 2, kv_len)
    press = EntropyGatedChunkKVPress(
        press=FixedScorer(compression_ratio=compression_ratio, scores=scores),
        chunk_length=chunk_length,
        rescue_size=4,
    )
    kept = run_press(scores, press)
    assert len(kept) == expected_budget(kv_len, compression_ratio)
    assert kept == sorted(set(kept)), "kept positions must be unique and in positional order"


def test_spiky_chunk_is_reduced_to_rescue_size():
    """An important but spiky chunk keeps only its needle, not all chunk_length tokens."""
    chunk_length, n_chunks, rescue_size = 10, 20, 1
    scores, needles = spiky_scores(n_chunks, chunk_length)
    kv_len = n_chunks * chunk_length
    # A budget of ~2 chunks: ChunkKV would spend it keeping 2 chunks whole, EG-ChunkKV rescues needles.
    press = EntropyGatedChunkKVPress(
        press=FixedScorer(compression_ratio=0.9, scores=scores),
        chunk_length=chunk_length,
        rescue_size=rescue_size,
        entropy_threshold=0.5,
    )
    kept = run_press(scores, press)
    assert len(kept) == expected_budget(kv_len, 0.9)

    # Every rescued needle survives, and the highest-scoring chunks are not kept whole.
    kept_set = set(kept)
    top_needles = sorted(needles, key=lambda i: -float(scores[0, 0, i]))[:20]
    assert kept_set.issuperset(top_needles[:10]), "the strongest needles must be retained"
    per_chunk = [len([k for k in kept if k // chunk_length == c]) for c in range(n_chunks)]
    assert max(per_chunk) < chunk_length, "no spiky chunk should be kept whole"


def test_entropy_threshold_degenerate_limits():
    """threshold=0 disables rescuing (chunks kept whole); threshold=1 rescues every important chunk."""
    chunk_length, n_chunks = 10, 20
    scores, _ = spiky_scores(n_chunks, chunk_length)
    kwargs = dict(chunk_length=chunk_length, rescue_size=1)

    never_spiky = EntropyGatedChunkKVPress(
        press=FixedScorer(compression_ratio=0.9, scores=scores), entropy_threshold=0.0, **kwargs
    )
    always_spiky = EntropyGatedChunkKVPress(
        press=FixedScorer(compression_ratio=0.9, scores=scores), entropy_threshold=1.0, **kwargs
    )
    kept_whole = run_press(scores, never_spiky)
    kept_rescued = run_press(scores, always_spiky)

    # Both spend exactly the same budget
    budget = expected_budget(n_chunks * chunk_length, 0.9)
    assert len(kept_whole) == len(kept_rescued) == budget

    # With rescuing disabled the budget goes to whole chunks; with it enabled the same budget
    # is spread over strictly more chunks, which is the point of the press.
    chunks_whole = len({k // chunk_length for k in kept_whole})
    chunks_rescued = len({k // chunk_length for k in kept_rescued})
    assert chunks_whole == 2, "without rescuing, a 20-token budget buys exactly 2 whole chunks"
    assert chunks_rescued > chunks_whole


def test_compression_ratio_is_delegated_to_inner_press():
    """The wrapper exposes and forwards the inner ScorerPress's compression ratio."""
    inner = FixedScorer(compression_ratio=0.3, scores=torch.rand(1, 2, 64))
    press = EntropyGatedChunkKVPress(press=inner, chunk_length=8)
    assert press.compression_ratio == 0.3
    press.compression_ratio = 0.7
    assert inner.compression_ratio == 0.7


def test_requires_scorer_press():
    with pytest.raises(AssertionError):
        EntropyGatedChunkKVPress(press="not-a-press")  # type: ignore[arg-type]
