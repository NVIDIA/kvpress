# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from typing import Optional

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
    top rescue_size tokens, and the freed budget is spent on further chunks. The
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
    rescue_size : int, default=4
        Number of tokens kept from an important but spiky chunk.
    entropy_threshold : float or None, default=None
        Spikiness cutoff on the normalized within-chunk entropy, in [0, 1]. A chunk is
        spiky when its entropy falls below this value. If None, the per-example median
        entropy over all chunks is used.

    Notes
    -----
    Chunk and token selection is shared across heads and computed from batch element 0,
    the same convention as ChunkKVPress; it is intended for the batch-size-1 context
    compression performed by the kvpress pipeline. Token scores are assumed to be
    non-negative (as produced by e.g. SnapKVPress) and are clamped before the entropy
    is computed.
    """

    chunk_length: int = 10
    rescue_size: int = 4
    entropy_threshold: Optional[float] = None

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
        c = self.chunk_length

        # Head-summed, non-negative per-token scores (batch element 0).
        global_scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        tok = global_scores.sum(dim=1)[0].clamp(min=0).float()  # (kv_len,)

        budget = max(1, int(kv_len * (1 - self.press.compression_ratio)))
        if budget >= kv_len:
            return keys, values

        # 1. Per-chunk semantic score S and normalized entropy H_tilde.
        n_chunks = math.ceil(kv_len / c)
        bounds = [(i * c, min(i * c + c, kv_len)) for i in range(n_chunks)]
        n_complete = kv_len // c
        remaining_tokens = kv_len % c

        X = tok[: n_complete * c].view(n_complete, c)
        s_scores = X.mean(dim=1)
        if c > 1:
            p = X / (X.sum(dim=1, keepdim=True) + EPSILON)
            h = -(p * (p + EPSILON).log()).sum(dim=1)
            ht = (h / math.log(c)).clamp(0.0, 1.0)
        else:
            ht = torch.zeros(n_complete, device=tok.device)

        # The trailing partial chunk does not fit the reshape and is handled separately.
        if remaining_tokens > 0:
            ts = tok[n_complete * c :]
            s_tail = ts.mean().unsqueeze(0)
            if remaining_tokens >= 2:
                pr = ts / (ts.sum() + EPSILON)
                hr = -(pr * (pr + EPSILON).log()).sum()
                ht_tail = (hr / math.log(remaining_tokens)).clamp(0.0, 1.0).unsqueeze(0)
            else:
                ht_tail = torch.zeros(1, device=tok.device)
            s_scores = torch.cat([s_scores, s_tail])
            ht = torch.cat([ht, ht_tail])

        med = s_scores.median()
        if self.entropy_threshold is None:
            tau = ht.median()
        else:
            tau = torch.tensor(float(self.entropy_threshold), device=tok.device)

        # 2. Greedy pass over chunks in decreasing semantic score.
        important_all = (s_scores >= med).tolist()
        spiky_all = (ht < tau).tolist()
        keep = torch.zeros(kv_len, dtype=torch.bool, device=tok.device)
        for i in torch.argsort(s_scores, descending=True).tolist():
            if budget <= 0:
                break
            s, e = bounds[i]
            n_i = e - s
            ts = tok[s:e]

            if important_all[i] and spiky_all[i]:
                # Important but spiky: keep only the highest-scoring tokens of the chunk.
                r = min(self.rescue_size, budget, n_i)
                keep[torch.topk(ts, r).indices + s] = True
                budget -= r
            elif n_i <= budget:
                # Coherent chunk that fits in the remaining budget: keep it whole.
                keep[s:e] = True
                budget -= n_i
            else:
                # Last chunk to be considered: keep as much of it as the budget allows.
                keep[torch.topk(ts, budget).indices + s] = True
                budget = 0

        # 3. Reducing spiky chunks may leave budget unspent. Top up with the highest-scoring
        # remaining tokens so that exactly (1 - compression_ratio) * kv_len tokens are kept.
        if budget > 0:
            leftover = (~keep).nonzero(as_tuple=False).squeeze(-1)
            if leftover.numel() > 0:
                add = min(budget, leftover.numel())
                keep[leftover[torch.topk(tok[leftover], add).indices]] = True

        # 4. Gather the retained keys and values in positional order.
        indices = keep.nonzero(as_tuple=False).squeeze(-1).sort()[0]
        indices = indices.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values
