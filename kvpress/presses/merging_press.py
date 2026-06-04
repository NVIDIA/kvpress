# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

# Epsilon for numerical stability — safe for float16 (min ~6e-8) and bfloat16
_EPS = 1e-6


@dataclass
class MergingPress(BasePress):
    """
    Merge-on-evict wrapper for any :class:`ScorerPress`.

    Replaces hard eviction with weighted value blending: each evicted token's
    value is folded into its most cosine-similar surviving neighbor, scaled by
    the relative value-norm of evictor and target. Keys are preserved (RoPE-safe).

    Inspired by Token Merging (Bolya et al., ICLR 2023, https://arxiv.org/abs/2210.09461)
    and D2O (Wan et al., 2024, https://arxiv.org/abs/2406.13035).

    🤖 automated agent contribution

    Parameters
    ----------
    press : ScorerPress
        Underlying scorer that decides which tokens survive.
    similarity_threshold : float, default=0.0
        Minimum cosine similarity for a merge to proceed.
    merge_fraction : float, default=1.0
        Fraction of evicted tokens (ranked by similarity) that are merged.
        Task-dependent: 1.0 wins on retrieval, 0.75 wins on extraction.
    """

    press: ScorerPress = None  # type: ignore[assignment]
    similarity_threshold: float = 0.0
    merge_fraction: float = 1.0

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), (
            f"MergingPress requires a ScorerPress, got {type(self.press).__name__}"
        )
        assert 0.0 <= self.similarity_threshold <= 1.0
        assert 0.0 < self.merge_fraction <= 1.0, "merge_fraction must be in (0, 1]"

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self) -> float:
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value: float) -> None:
        self.press.compression_ratio = value

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Identical to :meth:`ScorerPress.compress` except for the single line
        ``keys, values = self.merge(keys, values, indices)`` inserted between
        the top-k selection and the gather.
        """
        if self.press.compression_ratio == 0:
            return keys, values

        # Compute scores
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        # Get indices of KV pairs with the lowest scores
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.press.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices

        # Merge evicted tokens into the survivors before pruning
        keys, values = self.merge(keys, values, indices)

        # Prune keys and values
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values

    def merge(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fold each evicted token's value into its most cosine-similar survivor.

        Vectorized across (batch, head); writes the merged values back to the
        kept positions of the returned tensor. Keys are returned unchanged.

        Parameters
        ----------
        keys, values : Tensor, shape ``(B, H, L, D)``
        indices : Tensor, shape ``(B, H, n_kept)``
            Kept-position indices (output of ``scores.topk``).
        """
        bsz, num_heads, seq_len, head_dim = keys.shape
        n_kept = indices.shape[2]
        n_evict = seq_len - n_kept
        if n_evict == 0 or n_kept == 0:
            return keys, values

        # Derive evict indices as the complement of the kept indices
        evict_mask = torch.ones(bsz, num_heads, seq_len, device=keys.device, dtype=torch.bool)
        evict_mask.scatter_(2, indices, False)
        evict_idx = evict_mask.nonzero(as_tuple=False)[:, 2].reshape(bsz, num_heads, n_evict)

        keep_idx = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        evict_idx = evict_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        kept_keys = keys.gather(2, keep_idx).float()
        evict_keys = keys.gather(2, evict_idx).float()
        kept_values = values.gather(2, keep_idx)
        evict_values = values.gather(2, evict_idx)

        # Cosine similarity → nearest survivor (per evicted token, batched over B, H)
        kept_keys = kept_keys / kept_keys.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        evict_keys = evict_keys / evict_keys.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        max_sim, target = (evict_keys @ kept_keys.transpose(-2, -1)).max(dim=-1)

        # Threshold gate
        merge_ok = max_sim >= self.similarity_threshold

        # Fraction gate: keep only the top merge_fraction by similarity.
        # Failed tokens sink to -inf so the quantile lands among valid candidates.
        if self.merge_fraction < 1.0 and merge_ok.any():
            threshold = max_sim.masked_fill(~merge_ok, float("-inf")).quantile(
                1.0 - self.merge_fraction, dim=-1, keepdim=True
            )
            merge_ok = merge_ok & (max_sim >= threshold)

        # Similarity- and value-norm-weighted merge
        weights = max_sim.clamp(min=0) * merge_ok.float()
        target_norm = kept_values.float().norm(dim=-1).gather(2, target)
        evict_norm = evict_values.float().norm(dim=-1)
        weights = weights * evict_norm / (evict_norm + target_norm + _EPS)

        target = target.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        weights = weights.unsqueeze(-1)

        # Scatter-add evicted values into their kept-position targets (fp32 accumulation)
        value_accum = torch.zeros(bsz, num_heads, n_kept, head_dim, device=keys.device, dtype=torch.float32)
        value_accum.scatter_add_(2, target, weights * evict_values.float())

        weight_accum = torch.zeros(bsz, num_heads, n_kept, device=keys.device, dtype=torch.float32)
        weight_accum.scatter_add_(2, target[..., 0], weights[..., 0])

        # Normalized merge: only update positions that received any contribution
        kept_values = torch.where(
            (weight_accum > 0).unsqueeze(-1),
            ((kept_values.float() + value_accum) / (1.0 + weight_accum).unsqueeze(-1)).to(values.dtype),
            kept_values,
        )

        result_values = values.clone()
        result_values.scatter_(2, keep_idx, kept_values)
        return keys, result_values
