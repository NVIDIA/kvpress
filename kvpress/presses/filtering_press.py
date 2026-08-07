# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.padded_tensor import PaddedTensor
from kvpress.presses.decoding_press import DecodingPress


@dataclass
class FilteringPress(DecodingPress):
    """
    A decoding press that filters tokens during decoding by making online keep/skip decisions.

    Instead of retroactive eviction (scoring all tokens and removing the lowest-scored),
    this press decides for each new decode token whether to keep it in the cache.
    Only the newest token can be removed — existing cache entries are never modified.

    This makes the press compatible with append-only cache architectures (e.g. vLLM's
    paged KV cache). During prefill, this press is a no-op — filtering only applies
    to the decode phase, where tokens arrive one at a time and the cache is append-only.

    The decision is made per head: each head independently scores all tokens
    (including the new one) using the wrapped ScorerPress and checks whether the
    new token's score is above the eviction threshold at the target compression
    ratio. Rejected heads mark the position as padding; accepted heads that find
    an earlier padding slot move the token there to keep valid tokens packed.
    When all heads have padding at the last position, it is removed to shrink
    the cache.

    This press requires logical ``position_ids`` to be passed through the model
    forward call.

    Parameters
    ----------
    base_press : ScorerPress
        The scorer press used to compute importance scores for tokens.
    target_compression_ratio : float, default=0.5
        Target fraction of tokens to filter out during decoding.
    compression_interval : int, default=1
        Number of decoding steps between filtering decisions.
    fill_padding : bool, default=True
        Zero out padding positions after filtering. Per-head rejection creates
        positions that are padding for some heads but valid for others; zeroing
        prevents stale data from affecting attention. Disabling is faster but
        may degrade quality.
    hidden_states_buffer_size : int, default=256
        Maximum number of hidden states to keep before compression.
    """

    target_compression_ratio: float = 0.5
    compression_interval: int = 1
    fill_padding: bool = True
    target_size: int = field(default=1, init=False)

    def __post_init__(self):
        super().__post_init__()
        assert 0 <= self.target_compression_ratio < 1, "target_compression_ratio must be between 0 and 1"
        self._lengths = {}

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_tokens_seen = int(kwargs["position_ids"].max().item()) + 1
        n_kept = max(1, int(total_tokens_seen * (1 - self.target_compression_ratio)))
        n_kept = min(n_kept, keys.shape[2])

        layer_idx = getattr(module, "layer_idx", 0)
        if layer_idx in self._lengths:
            lengths = self._lengths[layer_idx]
        else:
            lengths = torch.full(keys.shape[:2], keys.shape[2] - 1, dtype=torch.long, device=keys.device)

        kt = PaddedTensor(keys.clone(), lengths.clone())
        vt = PaddedTensor(values.clone(), lengths.clone())
        valid_mask = kt.valid_mask(include_last=True)

        scores = self.base_press.score(
            module, hidden_states, kt.data, vt.data, attentions, {**kwargs, "valid_mask": valid_mask}
        )
        scores[~valid_mask] = float("-inf")

        threshold = scores.topk(n_kept, dim=-1, sorted=True).values[:, :, -1]
        rejected = scores[:, :, -1] < threshold

        if rejected.all():
            return keys[:, :, :-1, :].contiguous(), values[:, :, :-1, :].contiguous()

        kt.accept_last(~rejected)
        vt.accept_last(~rejected)
        if self.fill_padding:
            kt.fill_padding()
            vt.fill_padding()
        kt.shrink()
        vt.shrink()

        self._lengths[layer_idx] = kt.lengths.clone()
        return kt.data, vt.data

    def reset(self):
        super().reset()
        self._lengths = {}
