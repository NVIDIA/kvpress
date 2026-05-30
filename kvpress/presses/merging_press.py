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
    value is folded into its most cosine-similar surviving neighbor. Keys are
    preserved by default (RoPE-safe).

    Parameters
    ----------
    press : ScorerPress
        Underlying scorer that decides which tokens survive.
    similarity_threshold : float, default=0.0
        Minimum cosine similarity for a merge to proceed.
    merge_keys : bool, default=False
        Also merge evicted keys.  ``False`` preserves RoPE.
    value_norm_weighting : bool, default=True
        Scale merge weight by relative value-vector L2 norm.
    max_merge_per_token : int, default=0
        Cap on merges per survivor.  ``0`` disables.
    merge_fraction : float, default=1.0
        Fraction of evicted tokens (ranked by similarity) that are merged.
    """

    press: ScorerPress = None  # type: ignore[assignment]
    similarity_threshold: float = 0.0
    merge_keys: bool = False
    value_norm_weighting: bool = True
    max_merge_per_token: int = 0
    merge_fraction: float = 1.0

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), (
            f"MergingPress requires a ScorerPress, got {type(self.press).__name__}"
        )
        assert 0.0 <= self.similarity_threshold <= 1.0
        assert self.max_merge_per_token >= 0, "max_merge_per_token must be non-negative"
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

        Operates on the full tensors; only positions in ``indices`` are updated.
        Other positions are unchanged — they get pruned by :meth:`compress`
        after this returns.

        Parameters
        ----------
        keys, values : Tensor, shape ``(B, H, L, D)``
        indices : Tensor, shape ``(B, H, n_kept)``
            Kept-position indices (output of ``scores.topk``).
        """
        bsz, num_kv_heads, k_len, head_dim = keys.shape
        device = keys.device

        # Build evict mask from kept indices (True = evict)
        evict_mask = torch.ones(bsz, num_kv_heads, k_len, device=device, dtype=torch.bool)
        evict_mask.scatter_(2, indices, False)

        merged_values = values.float().clone()
        merged_keys = keys.float().clone() if self.merge_keys else None

        for b in range(bsz):
            for h in range(num_kv_heads):
                evict_idx = evict_mask[b, h].nonzero(as_tuple=True)[0]
                keep_idx = (~evict_mask[b, h]).nonzero(as_tuple=True)[0]
                n_evict = evict_idx.shape[0]
                n_kept = keep_idx.shape[0]

                if n_evict == 0 or n_kept == 0:
                    continue

                # Cosine similarity → nearest survivor (per evicted token)
                evict_k = keys[b, h, evict_idx].float()
                kept_k = keys[b, h, keep_idx].float()
                e_norms = evict_k.norm(dim=-1, keepdim=True).clamp(min=_EPS)
                k_norms = kept_k.norm(dim=-1, keepdim=True).clamp(min=_EPS)
                sim = (evict_k / e_norms) @ (kept_k / k_norms).T
                max_sim, target = sim.max(dim=-1)

                # Threshold gate
                merge_ok = max_sim >= self.similarity_threshold

                # Fraction gate: keep only the top merge_fraction by similarity
                if self.merge_fraction < 1.0 and merge_ok.any():
                    masked_sim = max_sim.clone()
                    masked_sim[~merge_ok] = float("inf")
                    q = 1.0 - self.merge_fraction
                    frac_threshold = masked_sim.quantile(q)
                    merge_ok = merge_ok & (max_sim >= frac_threshold)

                if not merge_ok.any():
                    continue

                # Similarity-weighted merge
                evict_v = values[b, h, evict_idx].float()
                w = max_sim.clamp(min=0) * merge_ok.float()

                if self.value_norm_weighting:
                    target_v = values[b, h, keep_idx[target]].float()
                    ev_norm = evict_v.norm(dim=-1)
                    tv_norm = target_v.norm(dim=-1)
                    w = w * ev_norm / (ev_norm + tv_norm + _EPS)

                # Merge-count cap: prevent survivor dilution
                if self.max_merge_per_token > 0:
                    count = torch.zeros(n_kept, device=device, dtype=torch.float32)
                    count.scatter_add_(0, target, merge_ok.float())
                    excess = (count / self.max_merge_per_token).clamp(min=1.0)
                    w = w / excess[target]

                w_exp = w.unsqueeze(-1)

                val_accum = torch.zeros(n_kept, head_dim, device=device, dtype=torch.float32)
                val_accum.scatter_add_(0, target.unsqueeze(-1).expand_as(evict_v), w_exp * evict_v)
                w_accum = torch.zeros(n_kept, device=device, dtype=torch.float32)
                w_accum.scatter_add_(0, target, w)

                active = w_accum > 0
                total_w = (1.0 + w_accum).unsqueeze(-1)

                orig_v = merged_values[b, h, keep_idx]
                new_v = (orig_v + val_accum) / total_w
                merged_values[b, h, keep_idx] = torch.where(active.unsqueeze(-1), new_v, orig_v)

                if self.merge_keys and merged_keys is not None:
                    evict_k_orig = keys[b, h, evict_idx].float()
                    key_accum = torch.zeros(n_kept, head_dim, device=device, dtype=torch.float32)
                    key_accum.scatter_add_(0, target.unsqueeze(-1).expand_as(evict_k_orig), w_exp * evict_k_orig)
                    orig_k = merged_keys[b, h, keep_idx]
                    new_k = (orig_k + key_accum) / total_w
                    merged_keys[b, h, keep_idx] = torch.where(active.unsqueeze(-1), new_k, orig_k)

        result_values = merged_values.to(values.dtype)
        result_keys = merged_keys.to(keys.dtype) if self.merge_keys else keys
        return result_keys, result_values
