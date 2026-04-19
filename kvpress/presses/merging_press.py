# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)

# Epsilon for numerical stability — safe for float16 (min ~6e-8) and bfloat16
_EPS = 1e-6


def _merge_on_evict(
    keys: torch.Tensor,
    values: torch.Tensor,
    evict_mask: torch.Tensor,
    similarity_threshold: float,
    merge_keys: bool,
    value_norm_weighting: bool,
    max_merge_per_token: int = 0,
    merge_fraction: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge evicted tokens into their most cosine-similar survivors.

    Handles variable per-head eviction counts (from AdaKV, DMSPress, or uniform
    scorer-based eviction).  Merged information is written in-place into the
    survivor positions of the full-length tensors.

    **Per-token value-reconstruction bound.**  For a single evicted token *i*
    routed to survivor *j* with cosine similarity :math:`w = \\cos(k_i, k_j)`:

    .. math::

        \\|\\Delta v_{\\text{merge}}\\| \\leq \\frac{1}{1 + w} \\;\\|v_i\\|

    This bounds per-token reconstruction error in value space only, not the
    end-to-end output after softmax re-normalization.

    Parameters
    ----------
    keys : Tensor, shape ``(B, H, L, D)``
    values : Tensor, shape ``(B, H, L, D)``
    evict_mask : Tensor, shape ``(B, H, L)``, dtype bool
        ``True`` at positions to evict, ``False`` at positions to keep.
    similarity_threshold : float
        Minimum cosine similarity for a merge to proceed.
    merge_keys : bool
        Whether to merge evicted information into survivor keys.
    value_norm_weighting : bool
        Scale merge weight by relative value-vector L2 norm.
    max_merge_per_token : int, default=0
        Cap on merges per survivor.  ``0`` disables.
    merge_fraction : float, default=1.0
        Fraction of evicted tokens (ranked by similarity) that are merged.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(new_keys, new_values)`` — same shape as input.

    References
    ----------
    .. [1] Bolya et al., "Token Merging: Your ViT But Faster", ICLR 2023.
    .. [2] Wan et al., "D2O: Dynamic Discriminative Operations", 2024.
    .. [3] Huang et al., "KeepKV: Lossless KV Cache Compression", 2025.
    """
    bsz, num_kv_heads, k_len, head_dim = keys.shape
    device = keys.device

    merged_values = values.float().clone()
    merged_keys = keys.float().clone() if merge_keys else None

    for b in range(bsz):
        for h in range(num_kv_heads):
            evict_idx = evict_mask[b, h].nonzero(as_tuple=True)[0]
            keep_idx = (~evict_mask[b, h]).nonzero(as_tuple=True)[0]
            n_evict = evict_idx.shape[0]
            n_kept = keep_idx.shape[0]

            if n_evict == 0 or n_kept == 0:
                continue

            # Cosine similarity → nearest survivor
            evict_k = keys[b, h, evict_idx].float()
            kept_k = keys[b, h, keep_idx].float()
            e_norms = evict_k.norm(dim=-1, keepdim=True).clamp(min=_EPS)
            k_norms = kept_k.norm(dim=-1, keepdim=True).clamp(min=_EPS)
            sim = (evict_k / e_norms) @ (kept_k / k_norms).T
            max_sim, target = sim.max(dim=-1)

            # Threshold gate
            merge_ok = max_sim >= similarity_threshold

            # Fraction gate: keep only top merge_fraction of evicted tokens
            if merge_fraction < 1.0 and merge_ok.any():
                masked_sim = max_sim.clone()
                masked_sim[~merge_ok] = float("inf")
                q = 1.0 - merge_fraction
                frac_threshold = masked_sim.quantile(q)
                merge_ok = merge_ok & (max_sim >= frac_threshold)

            if not merge_ok.any():
                continue

            # Similarity-weighted merge
            w = max_sim.clamp(min=0) * merge_ok.float()

            if value_norm_weighting:
                evict_v = values[b, h, evict_idx].float()
                target_v = values[b, h, keep_idx[target]].float()
                ev_norm = evict_v.norm(dim=-1)
                tv_norm = target_v.norm(dim=-1)
                w = w * ev_norm / (ev_norm + tv_norm + _EPS)

            # Merge count cap: prevent survivor dilution
            if max_merge_per_token > 0:
                count = torch.zeros(n_kept, device=device, dtype=torch.float32)
                count.scatter_add_(0, target, merge_ok.float())
                excess = (count / max_merge_per_token).clamp(min=1.0)
                w = w / excess[target]

            w_exp = w.unsqueeze(-1)
            evict_v = values[b, h, evict_idx].float()

            val_accum = torch.zeros(n_kept, head_dim, device=device, dtype=torch.float32)
            val_accum.scatter_add_(0, target.unsqueeze(-1).expand_as(evict_v), w_exp * evict_v)
            w_accum = torch.zeros(n_kept, device=device, dtype=torch.float32)
            w_accum.scatter_add_(0, target, w)

            active = w_accum > 0
            total_w = (1.0 + w_accum).unsqueeze(-1)

            orig_v = merged_values[b, h, keep_idx]
            new_v = (orig_v + val_accum) / total_w
            merged_values[b, h, keep_idx] = torch.where(active.unsqueeze(-1), new_v, orig_v)

            if merge_keys and merged_keys is not None:
                evict_k_orig = keys[b, h, evict_idx].float()
                key_accum = torch.zeros(n_kept, head_dim, device=device, dtype=torch.float32)
                key_accum.scatter_add_(0, target.unsqueeze(-1).expand_as(evict_k_orig), w_exp * evict_k_orig)
                orig_k = merged_keys[b, h, keep_idx]
                new_k = (orig_k + key_accum) / total_w
                merged_keys[b, h, keep_idx] = torch.where(active.unsqueeze(-1), new_k, orig_k)

    result_values = merged_values.to(values.dtype)
    result_keys = merged_keys.to(keys.dtype) if merge_keys else keys
    return result_keys, result_values


@dataclass
class MergingPress(BasePress):
    """
    Press-agnostic merge-on-evict wrapper for KV cache compression.

    Wraps a :class:`ScorerPress` and replaces hard eviction with merge-on-evict:
    each evicted token is folded into its most similar surviving neighbor rather than
    being discarded.  Values are blended via a similarity-weighted average; keys can
    optionally be merged depending on the ``merge_keys`` flag.

    Parameters
    ----------
    press : BasePress
        The underlying press.
    similarity_threshold : float, default=0.0
        Minimum cosine similarity for a merge to proceed.
    merge_keys : bool, default=False
        Whether to merge evicted keys.  ``False`` preserves RoPE encoding.
    value_norm_weighting : bool, default=True
        Scale merge weight by relative value-vector L2 norm.
    max_merge_per_token : int, default=0
        Cap on merges per survivor.  ``0`` disables.
    merge_fraction : float, default=1.0
        Fraction of evicted tokens (by similarity) that are merged.

    See also
    --------
    Bolya et al., "Token Merging", ICLR 2023.
    Zhang et al., "CaM", ICML 2024.
    Wan et al., "KeepKV", 2025.
    """

    press: BasePress
    similarity_threshold: float = 0.0
    merge_keys: bool = False
    value_norm_weighting: bool = True
    max_merge_per_token: int = 0
    merge_fraction: float = 1.0

    def __post_init__(self):
        assert isinstance(self.press, BasePress), f"MergingPress requires a BasePress, got {type(self.press)}"
        assert 0.0 <= self.similarity_threshold <= 1.0
        assert self.max_merge_per_token >= 0, "max_merge_per_token must be non-negative"
        assert 0.0 < self.merge_fraction <= 1.0, "merge_fraction must be in (0, 1]"

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def _merge_kwargs(self) -> dict:
        """Common kwargs for _merge_on_evict calls."""
        return dict(
            similarity_threshold=self.similarity_threshold,
            merge_keys=self.merge_keys,
            value_norm_weighting=self.value_norm_weighting,
            max_merge_per_token=self.max_merge_per_token,
            merge_fraction=self.merge_fraction,
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

        if not isinstance(self.press, ScorerPress):
            return self.press.compress(module, hidden_states, keys, values, attentions, kwargs)

        return self._compress_scorer(module, hidden_states, keys, values, attentions, kwargs)

    def _compress_scorer(self, module, hidden_states, keys, values, attentions, kwargs):
        bsz, num_kv_heads, k_len, head_dim = keys.shape
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        n_kept = int(k_len * (1 - self.press.compression_ratio))
        if n_kept >= k_len:
            return keys, values
        if n_kept <= 0:
            return keys[:, :, :0, :].contiguous(), values[:, :, :0, :].contiguous()

        # Build evict mask from scores, merge in-place, then extract kept positions
        keep_idx = scores.topk(n_kept, dim=-1).indices
        evict_mask = torch.ones(bsz, num_kv_heads, k_len, device=keys.device, dtype=torch.bool)
        evict_mask.scatter_(2, keep_idx, False)

        new_keys, new_values = _merge_on_evict(keys, values, evict_mask, **self._merge_kwargs())

        # Extract kept positions (topk order to match ScorerPress.compress behavior)
        idx4 = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        return new_keys.gather(2, idx4), new_values.gather(2, idx4)
