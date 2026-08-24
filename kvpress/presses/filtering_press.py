# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch

from kvpress.presses.decoding_press import DecodingPress


@dataclass
class FilteringPress(DecodingPress):
    """
    A decoding press that filters tokens during decoding by making online keep/skip decisions.

    Instead of retroactive eviction (scoring all tokens and removing the lowest-scored),
    this press decides for each new decode token whether to keep it in the cache.
    Only the newest token can be removed — existing cache entries are never modified.

    The decision model is compatible with append-only caches - each step's decision
    depends only on the newest token - but this reference implementation still writes
    back a full tensor.

    The decision is made per head: each head independently scores all valid tokens
    (including the new one) using the wrapped ScorerPress and checks whether the
    new token's score is above the eviction threshold at the target compression
    ratio. Rejected positions are masked via ``masked_key_indices``.
    When all heads reject a token, it is physically removed from the cache.

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
    hidden_states_buffer_size : int, default=256
        Maximum number of hidden states to keep before compression.
    """

    target_compression_ratio: float = 0.5
    compression_interval: int = 1
    target_size: int = field(default=1, init=False)

    def __post_init__(self):
        super().__post_init__()
        assert 0 <= self.target_compression_ratio < 1, "target_compression_ratio must be between 0 and 1"

    def compress(
        self,
        module,
        hidden_states,
        keys,
        values,
        attentions,
        kwargs,
    ):
        bsz, n_heads, k_len, _ = keys.shape
        position_ids = kwargs["position_ids"]
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[0] == 1:
            position_ids = position_ids.expand(bsz, -1)
        total_tokens_seen = position_ids.max(dim=-1).values + 1
        n_kept = (total_tokens_seen.float() * (1 - self.target_compression_ratio)).round().long()
        n_kept = n_kept.clamp(min=1, max=k_len)

        if (n_kept >= k_len).all():
            return keys, values

        # Build per-head valid mask from accumulated masked_key_indices
        valid_mask = torch.ones(bsz, n_heads, k_len, dtype=torch.bool, device=keys.device)
        masked: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = getattr(module, "masked_key_indices", None)
        if masked is not None:
            b_idx, h_idx, s_idx = masked
            valid_mask[b_idx, h_idx, s_idx] = False
        valid_mask[:, :, -1] = True

        # Score per head using valid_mask
        scores = torch.full((bsz, n_heads, k_len), float("-inf"), device=keys.device, dtype=keys.dtype)
        for b in range(bsz):
            for h in range(n_heads):
                valid_pos = valid_mask[b, h].nonzero(as_tuple=True)[0]
                head_keys = keys[b : b + 1, h : h + 1, valid_pos, :]
                head_values = values[b : b + 1, h : h + 1, valid_pos, :]
                head_scores = self.base_press.score(
                    module, hidden_states[b : b + 1], head_keys, head_values, attentions, kwargs
                )
                scores[b, h, valid_pos] = head_scores[0, 0]

        sorted_scores, _ = scores.sort(dim=-1, descending=True)
        idx = (n_kept - 1).view(-1, 1, 1).expand(-1, n_heads, 1)
        threshold = sorted_scores.gather(-1, idx).squeeze(-1)
        rejected = scores[:, :, -1] < threshold

        if rejected.all():
            return keys[:, :, :-1, :].contiguous(), values[:, :, :-1, :].contiguous()

        if not rejected.any():
            return keys, values

        new_b, new_h = rejected.nonzero(as_tuple=True)
        new_s = torch.full_like(new_b, k_len - 1)
        old: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = getattr(module, "masked_key_indices", None)
        if old is not None:
            new_b = torch.cat([old[0], new_b])
            new_h = torch.cat([old[1], new_h])
            new_s = torch.cat([old[2], new_s])
        module.masked_key_indices = (new_b, new_h, new_s)

        return keys, values
