# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values, get_prerope_query_states

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _aggregate_attention_per_kv_head(
    attentions: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Average attention scores across query heads that share a KV head."""
    num_query_heads = attentions.shape[1]
    if num_query_heads == num_kv_heads:
        return attentions
    group_size = num_query_heads // num_kv_heads
    batch, _, seq_q, seq_k = attentions.shape
    return attentions.reshape(batch, num_kv_heads, group_size, seq_q, seq_k).mean(dim=2)


@dataclass
class CAMPress(DecodingPress):
    """
    Cache Merging (CaM) KV cache compression during decoding.

    Evicted tokens' values are merged into their sequential neighbors using a
    Bernoulli merge probability derived from relative attention scores. Keys are
    pruned after merging.

    Based on CaM (https://openreview.net/forum?id=LCTmppB165).

    Parameters
    ----------
    base_press : ScorerPress
        Scorer used to select which tokens to evict (e.g., StreamingLLMPress).
    compression_ratio : float, default=0.0
        Fraction of prefill tokens to evict during decoding.
    merge_budget : int or None, default=64
        Number of sequential neighbors to merge each evicted token into.
        None merges into all remaining tokens after the evicted position.
    use_triton : bool, default=True
        Use the Triton kernel for merging when available (CUDA only).
    """

    base_press: ScorerPress = None
    compression_ratio: float = 0.0
    merge_budget: Optional[int] = 64
    use_triton: bool = True

    def __init__(
        self,
        base_press: ScorerPress,
        compression_ratio: float = 0.0,
        merge_budget: Optional[int] = 64,
        use_triton: bool = True,
    ):
        self.base_press = base_press
        self.compression_ratio = compression_ratio
        self.merge_budget = merge_budget
        self.use_triton = use_triton
        self._target_cache_size: dict[int, int] = {}
        self._first_eviction_done: dict[int, bool] = defaultdict(lambda: False)

    def post_init_from_model(self, model: PreTrainedModel):
        if hasattr(self.base_press, "post_init_from_model"):
            self.base_press.post_init_from_model(model)

    def reset(self):
        """Reset per-sequence state."""
        self._target_cache_size = {}
        self._first_eviction_done = defaultdict(lambda: False)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: Optional[torch.Tensor],
        kwargs: dict,
    ) -> torch.Tensor:
        """Delegate scoring to base_press with the compression ratio adjusted for the current cache size."""
        cache_len = keys.shape[2]
        n_to_evict = cache_len - self._target_cache_size[int(module.layer_idx)]
        cr = n_to_evict / cache_len if cache_len > 0 else 0.0

        old_cr = self.base_press.compression_ratio
        self.base_press.compression_ratio = cr
        try:
            scores = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)
        finally:
            self.base_press.compression_ratio = old_cr

        return scores

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: Optional[torch.Tensor],
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge evicted token values into sequential neighbors, then prune."""
        layer_idx = int(module.layer_idx)
        head_dim = module.head_dim
        cache_len = keys.shape[2]

        target_size = self._target_cache_size[layer_idx]
        n_to_evict = cache_len - target_size

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        batch, kv_heads, _ = scores.shape
        dev = scores.device
        n_kept = cache_len - n_to_evict

        kept_indices = scores.topk(n_kept, dim=-1).indices
        kept_indices = torch.sort(kept_indices, dim=-1).values

        all_idx = torch.arange(cache_len, device=dev)
        kept_mask = torch.zeros(batch, kv_heads, cache_len, dtype=torch.bool, device=dev)
        kept_mask.scatter_(2, kept_indices, True)
        evicted_positions = all_idx.expand(batch, kv_heads, -1)[~kept_mask].reshape(batch, kv_heads, n_to_evict)

        effective_budget = self.merge_budget if self.merge_budget is not None else (cache_len - 1)
        actual_budget = min(effective_budget, cache_len - 1)

        offsets = torch.arange(actual_budget, device=dev)
        per_token_targets = (evicted_positions.unsqueeze(-1) + 1 + offsets).clamp(max=cache_len - 1)
        valid_targets = (evicted_positions.unsqueeze(-1) + 1 + offsets) < cache_len

        merge_mask = None

        if attentions is None and actual_budget > 0 and n_to_evict > 0:
            attentions = self._compute_current_token_attention(module, hidden_states, keys, kwargs)

        if attentions is not None and actual_budget > 0 and n_to_evict > 0:
            attn_per_kv = _aggregate_attention_per_kv_head(attentions, kv_heads)
            if attn_per_kv.shape[2] > 1:
                attn_per_kv = attn_per_kv[:, :, -1:, :]
            attn_squeezed = attn_per_kv.squeeze(2)

            evicted_attn = attn_squeezed.gather(2, evicted_positions)
            per_token_target_attn = (
                attn_squeezed.unsqueeze(2).expand(-1, -1, n_to_evict, -1).gather(3, per_token_targets)
            )
            per_token_target_attn = per_token_target_attn.masked_fill(
                ~valid_targets.expand_as(per_token_target_attn), float("-inf")
            )
            ref_attn = per_token_target_attn.max(dim=-1).values

            merge_prob = torch.where(
                ref_attn > 0,
                (evicted_attn.float() / ref_attn.float().clamp(min=1e-9)).clamp(0.0, 1.0),
                torch.zeros_like(evicted_attn, dtype=torch.float32),
            ).to(evicted_attn.dtype)
            merge_mask = torch.bernoulli(merge_prob)

            non_merge = merge_mask < 0.5
            if non_merge.any():
                b_idx = torch.arange(batch, device=dev)[:, None, None].expand_as(evicted_positions)
                h_idx = torch.arange(kv_heads, device=dev)[None, :, None].expand_as(evicted_positions)
                pos_to_zero = evicted_positions[non_merge]
                if pos_to_zero.numel() > 0:
                    values[b_idx[non_merge], h_idx[non_merge], pos_to_zero, :] = 0.0

            is_first = not self._first_eviction_done[layer_idx]
            n_merged = int(merge_mask.sum().item())
            logger.debug(
                f"CaM L{layer_idx}: {'BULK' if is_first else 'step'} evict={n_to_evict}, "
                f"merged={n_merged}/{n_to_evict}, mean_prob={merge_prob.mean():.3f}, "
                f"cache={cache_len}->{n_kept}"
            )
        else:
            logger.debug(f"CaM L{layer_idx}: no attention, always-merge, evict={n_to_evict}")

        if actual_budget > 0 and n_to_evict > 0:
            if n_to_evict == 1 and merge_mask is not None and merge_mask.sum() == 0:
                pass
            else:
                if not per_token_targets.is_contiguous():
                    per_token_targets = per_token_targets.contiguous()

                if not evicted_positions.is_contiguous():
                    evicted_positions = evicted_positions.contiguous()

                valid_targets_c = valid_targets if valid_targets.is_contiguous() else valid_targets.contiguous()

                if self.use_triton and HAS_TRITON and values.is_cuda:
                    values = self._triton_merge(
                        values, evicted_positions, per_token_targets, actual_budget, valid_targets_c
                    )
                else:
                    values = self._torch_merge(
                        values, evicted_positions, per_token_targets, actual_budget, valid_targets_c
                    )

        gather_idx = kept_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        keys = keys.gather(2, gather_idx).contiguous()
        values = values.gather(2, gather_idx).contiguous()

        return keys, values

    @staticmethod
    def _compute_current_token_attention(
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        """Compute softmax attention from the last query token to all cached keys."""
        _, num_kv_heads, cache_len, head_dim = keys.shape
        num_query_heads = module.config.num_attention_heads
        num_key_value_groups = num_query_heads // num_kv_heads

        query_states = get_prerope_query_states(module, hidden_states)
        query_states = query_states[:, :, -1:, :]

        cos, sin = kwargs["position_embeddings"]
        cos = cos[:, -1:, :].unsqueeze(1)
        sin = sin[:, -1:, :].unsqueeze(1)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)

        keys_repeated = repeat_kv(keys, num_key_value_groups)
        scores = torch.matmul(query_states, keys_repeated.transpose(-2, -1)) / math.sqrt(head_dim)
        return torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

    def forward_hook(
        self,
        module: nn.Module,
        input: list[torch.Tensor],
        kwargs: dict,
        output: list,
    ):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        layer_idx = int(module.layer_idx)

        # Only operate during decoding
        if kwargs["cache_position"][-1] <= q_len:
            return output

        if self.compression_ratio <= 0:
            return output

        keys, values = extract_keys_and_values(cache, layer_idx)
        cache_len = keys.shape[2]

        if layer_idx not in self._target_cache_size:
            prefill_len = cache_len - 1
            self._target_cache_size[layer_idx] = max(int(prefill_len * (1 - self.compression_ratio)), 1)

        if cache_len <= self._target_cache_size[layer_idx]:
            return output

        attentions = output[1] if len(output) > 1 and output[1] is not None else None

        keys, values = self.compress(module, hidden_states, keys, values, attentions, kwargs)

        cache.layers[layer_idx].keys = keys
        cache.layers[layer_idx].values = values
        self._first_eviction_done[layer_idx] = True

        return output

    def _torch_merge(self, values, evicted_positions, per_token_targets, actual_budget, valid_targets):
        """Merge each evicted token's value into its sequential neighbors (pure PyTorch fallback)."""
        n_evicted = evicted_positions.shape[2]

        for i in range(n_evicted):
            for b in range(values.shape[0]):
                for h in range(values.shape[1]):
                    p = evicted_positions[b, h, i].item()
                    v_evicted = values[b, h, p, :].clone()
                    if v_evicted.abs().sum() < 1e-12:
                        continue
                    contribution = v_evicted / actual_budget
                    for t in range(actual_budget):
                        if valid_targets[b, h, i, t].item():
                            target = per_token_targets[b, h, i, t].item()
                            values[b, h, target, :] += contribution

        return values

    def _triton_merge(self, values, evicted_positions, per_token_targets, actual_budget, valid_targets):
        """Merge each evicted token's value into its sequential neighbors (Triton kernel)."""
        if not HAS_TRITON:
            return self._torch_merge(values, evicted_positions, per_token_targets, actual_budget, valid_targets)

        batch_size, num_kv_heads, seq_len, head_dim = values.shape
        n_evicted = evicted_positions.shape[2]

        BLOCK_D = triton.next_power_of_2(head_dim)
        TILE_R = min(64, triton.next_power_of_2(actual_budget))
        grid = (batch_size, num_kv_heads)

        _cam_decoding_merge_kernel[grid](
            values_ptr=values,
            evicted_pos_ptr=evicted_positions,
            merge_targets_ptr=per_token_targets,
            seq_len=seq_len,
            n_evicted=n_evicted,
            actual_budget=actual_budget,
            v_stride_b=values.stride(0),
            v_stride_h=values.stride(1),
            v_stride_s=values.stride(2),
            v_stride_d=values.stride(3),
            ep_stride_b=evicted_positions.stride(0),
            ep_stride_h=evicted_positions.stride(1),
            ep_stride_e=evicted_positions.stride(2),
            mt_stride_b=per_token_targets.stride(0),
            mt_stride_h=per_token_targets.stride(1),
            mt_stride_e=per_token_targets.stride(2),
            mt_stride_t=per_token_targets.stride(3),
            head_dim=head_dim,
            BLOCK_D=BLOCK_D,
            TILE_R=TILE_R,
        )
        return values


if HAS_TRITON:

    @triton.jit
    def _cam_decoding_merge_kernel(
        values_ptr,
        evicted_pos_ptr,
        merge_targets_ptr,
        seq_len,
        n_evicted,
        actual_budget,
        v_stride_b,
        v_stride_h,
        v_stride_s,
        v_stride_d,
        ep_stride_b,
        ep_stride_h,
        ep_stride_e,
        mt_stride_b,
        mt_stride_h,
        mt_stride_e,
        mt_stride_t,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
        TILE_R: tl.constexpr,
    ):
        """
        Tiled scatter-add merge kernel. Grid: (batch_size, num_kv_heads).
        Each evicted token scatters its contribution into its own neighbor list.
        """
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        v_base = values_ptr + batch_idx * v_stride_b + head_idx * v_stride_h
        ep_base = evicted_pos_ptr + batch_idx * ep_stride_b + head_idx * ep_stride_h
        mt_base = merge_targets_ptr + batch_idx * mt_stride_b + head_idx * mt_stride_h

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim

        for evict_idx in tl.range(0, n_evicted):
            token_pos = tl.load(ep_base + evict_idx * ep_stride_e).to(tl.int64)
            v_evicted = tl.load(
                v_base + token_pos * v_stride_s + d_offsets * v_stride_d,
                mask=d_mask,
                other=0.0,
            )
            contribution = v_evicted / actual_budget

            mt_evict_base = mt_base + evict_idx * mt_stride_e
            n_tiles = (actual_budget + TILE_R - 1) // TILE_R

            for tile_idx in tl.range(0, n_tiles):
                t_offsets = tl.arange(0, TILE_R)
                t_indices = tile_idx * TILE_R + t_offsets
                t_mask = t_indices < actual_budget

                target_positions = tl.load(
                    mt_evict_base + t_indices * mt_stride_t,
                    mask=t_mask,
                    other=0,
                ).to(tl.int64)

                valid = (target_positions < seq_len) & t_mask
                ptrs = v_base + target_positions[:, None] * v_stride_s + d_offsets[None, :] * v_stride_d
                mask_2d = valid[:, None] & d_mask[None, :]

                v_block = tl.load(ptrs, mask=mask_2d, other=0.0)
                v_block = v_block + contribution[None, :]
                tl.store(ptrs, v_block, mask=mask_2d)
