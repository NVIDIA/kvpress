# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.chunkkv_press import ChunkKVPress

EPSILON = 1e-8


@dataclass
class EntropyGatedChunkKVPress(ChunkKVPress):
    """
    EntropyGatedChunkKV: chunk selection gated by within-chunk score entropy.

    Extends ChunkKVPress, which keeps or drops every chunk as a whole. A chunk whose
    importance comes from a single high-scoring token therefore spends chunk_length
    cache slots to preserve one useful token. This press measures the normalized
    entropy of the token scores inside each chunk: coherent chunks (high entropy) are
    kept whole, while important but spiky chunks (low entropy) are reduced to their
    top low_entropy_chunk_length tokens, and the freed budget is spent on further chunks. The
    number of retained tokens is exactly (1 - compression_ratio) * kv_len, matching
    the budget of ChunkKVPress.

    Based on ChunkKV (https://arxiv.org/abs/2502.00299).

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method used to compute global importance scores.
    chunk_length : int, default=10
        Length of each chunk for token selection. Shorter than the ChunkKVPress default
        of 20: a finer granularity gives the gate more chunks to reallocate budget
        between, which is where the gain comes from.
    low_entropy_chunk_length : int, default=4
        Number of tokens kept from an important but spiky chunk.

    Notes
    -----
    Chunk and token selection is shared across heads and computed from batch element 0,
    the same convention as ChunkKVPress; it is intended for the batch-size-1 context
    compression performed by the kvpress pipeline. Ranking and top-k selection use the
    raw scores, so signed scorers (e.g. KeyDiffPress) are ordered correctly; the entropy
    gate rebases negative chunks to form a valid distribution but is most meaningful for
    non-negative scores (e.g. SnapKVPress).
    """

    chunk_length: int = 10
    low_entropy_chunk_length: int = 4

    def __post_init__(self):
        super().__post_init__()
        assert self.chunk_length > self.low_entropy_chunk_length >= 1, (
            "EntropyGatedChunkKVPress requires chunk_length > low_entropy_chunk_length >= 1"
        )

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.press.compression_ratio == 0:
            return keys, values
        assert attentions is None, "EntropyGatedChunkKVPress does not support attentions."

        kv_len = keys.shape[2]
        chunk_len = self.chunk_length

        # Head-summed per-token scores (batch element 0), kept raw so ranking works for signed scorers.
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        scores = scores.sum(dim=1)[0].float()  # (kv_len,)

        budget = max(1, int(kv_len * (1 - self.press.compression_ratio)))

        # 1. Per-chunk score and entropy.
        n_chunks = math.ceil(kv_len / chunk_len)
        bounds = [(i * chunk_len, min(i * chunk_len + chunk_len, kv_len)) for i in range(n_chunks)]
        n_complete = kv_len // chunk_len
        remaining_tokens = kv_len % chunk_len

        chunk_token_scores = scores[: n_complete * chunk_len].view(n_complete, chunk_len)
        chunk_scores = chunk_token_scores.mean(dim=1)
        chunk_token_scores = chunk_token_scores - chunk_token_scores.amin(dim=1, keepdim=True).clamp(max=0.0)
        p = chunk_token_scores / (chunk_token_scores.sum(dim=1, keepdim=True) + EPSILON)
        h = -(p * (p + EPSILON).log()).sum(dim=1)
        chunk_entropy = (h / math.log(chunk_len)).clamp(0.0, 1.0)

        # The trailing partial chunk does not fit the reshape and is handled separately.
        if remaining_tokens > 0:
            tail_scores = scores[n_complete * chunk_len :]
            chunk_scores_tail = tail_scores.mean().unsqueeze(0)
            if remaining_tokens == 1:
                chunk_entropy_tail = torch.zeros(1, device=scores.device)
            else:
                if (tail_scores < 0).any():
                    tail_scores = tail_scores - tail_scores.min().clamp(max=0.0)
                pr = tail_scores / (tail_scores.sum() + EPSILON)
                hr = -(pr * (pr + EPSILON).log()).sum()
                chunk_entropy_tail = (hr / math.log(remaining_tokens)).clamp(0.0, 1.0).unsqueeze(0)
            chunk_scores = torch.cat([chunk_scores, chunk_scores_tail])
            chunk_entropy = torch.cat([chunk_entropy, chunk_entropy_tail])

        score_threshold = chunk_scores.median()
        entropy_threshold = chunk_entropy.median()

        # 2. Greedy pass over chunks in decreasing semantic score.
        high_score_chunks = (chunk_scores >= score_threshold).tolist()
        low_entropy_chunks = (chunk_entropy < entropy_threshold).tolist()
        keep = torch.zeros(kv_len, dtype=torch.bool, device=scores.device)
        for chunk_idx in torch.argsort(chunk_scores, descending=True).tolist():
            if budget <= 0:
                break

            start, end = bounds[chunk_idx]
            n_kept = min(end - start, budget)
            if high_score_chunks[chunk_idx] and low_entropy_chunks[chunk_idx]:
                # Important but spiky: keep only the highest-scoring tokens of the chunk.
                n_kept = min(n_kept, self.low_entropy_chunk_length)

            if n_kept == end - start:
                keep[start:end] = True
            else:
                top_indices = torch.topk(scores[start:end], n_kept).indices + start
                keep[top_indices] = True
            budget -= n_kept

        # 3. Reducing spiky chunks may leave budget unspent. Top up with the highest-scoring
        # remaining tokens so that exactly (1 - compression_ratio) * kv_len tokens are kept.
        if budget > 0:
            leftover = (~keep).nonzero(as_tuple=False).squeeze(-1)
            if leftover.numel() > 0:
                add = min(budget, leftover.numel())
                keep[leftover[torch.topk(scores[leftover], add).indices]] = True

        # 4. Gather the retained keys and values in positional order.
        indices = keep.nonzero(as_tuple=False).squeeze(-1).sort()[0]
        indices = indices.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values
