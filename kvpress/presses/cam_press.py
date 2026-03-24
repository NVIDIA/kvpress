# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import QuantizedCache
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.adakv_press import AdaKVPress
from kvpress.utils import extract_keys_and_values, get_prerope_query_states

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


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

    base_press: ScorerPress | AdaKVPress = None
    compression_interval: int = 2
    target_size: int = 3048
    hidden_states_buffer_size: int = 256
    merge_budget: int = 32
    use_triton: bool = False

    def __post_init__(self):
        assert isinstance(self.base_press, (ScorerPress, AdaKVPress)), "CAMPress requires a ScorerPress as base_press"
        assert self.compression_interval > 0, "compression_interval must be greater than 0"
        assert self.target_size > 0, "target_size must be greater than 0"
        assert self.merge_budget > 0, "merge_budget must be positive "
        assert isinstance(self.use_triton, bool), "use_triton must be a boolean"

        # State Variables
        self.layer_step_counts = defaultdict(int)
        self._running_attn_sum: dict[int, torch.Tensor] = {}

        if self.use_triton and not HAS_TRITON:
            logger.warning(
                f"Triton is not available. Falling back to PyTorch merge implementation for {self.__class__.__name__}."
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
        layer_idx = int(module.layer_idx)
        cache_len = keys.shape[2]

        n_to_evict = cache_len-self.target_size

        target_compression_ratio = self._find_target_compression_ratio(cache_len, self.target_size)
        

        if n_to_evict <= 0:
            return keys, values

        # Temporary override base press ratio to get correct topK scores
        old_cr = self.base_press.compression_ratio
        self.base_press.compression_ratio = target_compression_ratio
        scores = self.base_press.score(module, hidden_states, keys, values, None, kwargs)
        self.base_press.compression_ratio = old_cr

        bsz, num_key_value_heads, _, head_dim = keys.shape

        evict_indices = scores[:, 0, :].topk(n_to_evict, dim=-1, largest=False).indices
        evict_indices = torch.sort(evict_indices, dim=-1).values

        evict_scores = scores[:, 0, :].gather(-1, evict_indices)
        # Flip so later sequence positions come first; stable sort preserves this order for ties
        k = self.layer_step_counts[layer_idx]
        order = evict_scores.flip(-1).argsort(dim=-1, descending=True, stable=True)[:, :k]
        merge_indices = evict_indices.gather(-1, n_to_evict - 1 - order)
        merge_indices = torch.sort(merge_indices, dim=-1).values

        kept_indices = scores[:, 0, :].topk(self.target_size, dim=-1).indices
        kept_indices = torch.sort(kept_indices, dim=-1).values

        if n_to_evict > 0:
            if self.use_triton and HAS_TRITON and values.is_cuda:
                values = self._triton_merge(values, merge_indices, kept_indices, attentions, self.merge_budget)
            else:
                values = self._torch_merge(values, merge_indices, kept_indices, attentions, self.merge_budget)

        # Physical Pruning
        kept_indices_expand = kept_indices.view(bsz, 1, self.target_size, 1).expand(bsz, num_key_value_heads, self.target_size, head_dim)
        keys = keys.gather(2, kept_indices_expand).contiguous()
        values = values.gather(2, kept_indices_expand).contiguous()

        # prune cumulative attentions
        expanded_indices = kept_indices.unsqueeze(1).expand(bsz, num_key_value_heads, -1)
        self._running_attn_sum[layer_idx] = self._running_attn_sum[layer_idx].gather(2, expanded_indices).contiguous()

        return keys, values

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

        cache_layer = cache.layers[module.layer_idx]
        keys, values = extract_keys_and_values(cache, layer_idx)
        bsz, num_key_value_heads, seq_len, _ = keys.shape

        # Accumulate Cumulative Attention over generation steps
        attentions = output[1] if len(output) > 1 and output[1] is not None else None
        if attentions is None:
            attentions = self._compute_current_token_attention(module, hidden_states, keys, kwargs)
        else:
            attentions = attentions[:,:,-1:,:]

        attentions = self._aggregate_attention_per_kv_head(attentions, num_key_value_heads)

        if attentions is not None:
            attn_squeezed = attentions.squeeze(2)

            if layer_idx not in self._running_attn_sum:
                self._running_attn_sum[layer_idx] = attn_squeezed.clone()
            else:
                # Pad running sum for the new token growth
                prev_len = self._running_attn_sum[layer_idx].shape[-1]
                pad_len = seq_len - prev_len

                if pad_len > 0:
                    pad = torch.zeros(
                        (bsz, num_key_value_heads, pad_len), device=attn_squeezed.device, dtype=attn_squeezed.dtype
                    )
                    self._running_attn_sum[layer_idx] = torch.cat([self._running_attn_sum[layer_idx], pad], dim=-1)

                self._running_attn_sum[layer_idx] += attn_squeezed

        self.layer_step_counts[layer_idx] += 1

        # Trigger interval-based bulk eviction
        if (self.layer_step_counts[layer_idx] >= self.compression_interval and seq_len>self.target_size) or (q_len >= self.target_size):

            attn_squeezed = self._running_attn_sum[layer_idx]
            keys, values = self.compress(module, hidden_states, keys, values, attn_squeezed, kwargs)

            # Update cache with compressed keys and values
            if isinstance(cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
                cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
                cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
                cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
                cache_layer.cumulative_length = keys.shape[2]
            else:
                cache_layer.keys = keys
                cache_layer.values = values

            self.layer_step_counts[layer_idx] = 0

        return output

    def reset(self):
        """Reset per-sequence state."""
        self.layer_step_counts = defaultdict(int)
        self._running_attn_sum: dict[int, torch.Tensor] = {}

    @staticmethod
    def _compute_current_token_attention(
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        """Compute softmax attention from the last query token to all cached keys."""
        _, num_key_value_heads, cache_len, head_dim = keys.shape
        num_query_heads = module.config.num_attention_heads
        num_key_value_groups = num_query_heads // num_key_value_heads

        query_states = get_prerope_query_states(module, hidden_states)
        query_states = query_states[:, :, -1:, :]

        cos, sin = kwargs["position_embeddings"]
        cos = cos[:, -1:, :].unsqueeze(1)
        sin = sin[:, -1:, :].unsqueeze(1)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)

        keys_repeated = repeat_kv(keys, num_key_value_groups)
        scores = torch.matmul(query_states, keys_repeated.transpose(-2, -1)) / math.sqrt(head_dim)
        return torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

    @staticmethod
    def _aggregate_attention_per_kv_head(
        attentions: torch.Tensor,
        num_key_value_heads: int,
    ) -> torch.Tensor:
        """Average attention scores across query heads that share a KV head."""
        num_query_heads = attentions.shape[1]
        if num_query_heads == num_key_value_heads:
            return attentions
        group_size = num_query_heads // num_key_value_heads
        bsz, _, seq_q, seq_k = attentions.shape
        return attentions.reshape(bsz, num_key_value_heads, group_size, seq_q, seq_k).mean(dim=2)

    def _torch_merge(self, values, merge_indices, kept_indices, attentions_per_kv, merge_budget):
        bsz, num_kv_heads, seq_len, head_dim = values.shape
        n_merge = merge_indices.shape[1]
        n_kept = kept_indices.shape[1]

        # 1. Cascading target starts
        base_idx_first = torch.searchsorted(kept_indices, merge_indices[:, 0:1], right=True)
        target_starts = torch.arange(n_merge, device=kept_indices.device).unsqueeze(0) + base_idx_first

        # 2. Build target window indices: [bsz, n_merge, merge_budget]
        offsets = torch.arange(merge_budget, device=kept_indices.device)
        window_idx = target_starts.unsqueeze(-1) + offsets.view(1, 1, -1)
        valid_mask = window_idx < n_kept
        window_idx = window_idx.clamp(max=n_kept - 1)
        target_positions = kept_indices.gather(1, window_idx.view(bsz, -1)).view(bsz, n_merge, merge_budget)

        # 3. Actual budget per merge token
        actual_budget = valid_mask.sum(dim=-1)

        # 4. Suffix mean via cumsum
        attn_cumsum = torch.nn.functional.pad(attentions_per_kv.cumsum(dim=-1), (1, 0))
        total_sum = attn_cumsum[:, :, -1:]
        start_sum = attn_cumsum.gather(2, target_starts.unsqueeze(1).expand(-1, num_kv_heads, -1))
        suffix_sum = total_sum - start_sum
        suffix_len = (seq_len - target_starts).unsqueeze(1)
        mean_attn = suffix_sum / suffix_len

        # 5. Merge probability
        merge_token_attn = attentions_per_kv.gather(2, merge_indices.unsqueeze(1).expand(-1, num_kv_heads, -1))
        merge_prob = merge_token_attn / mean_attn
        merge_prob = torch.where(torch.isnan(merge_prob), torch.zeros_like(merge_prob), merge_prob)
        merge_prob = torch.where(torch.isinf(merge_prob), torch.ones_like(merge_prob), merge_prob)
        merge_prob = merge_prob.clamp(0, 1)

        # 6. Bernoulli sampling
        merge_mask = torch.bernoulli(merge_prob)

        # 7. Build contributions and scatter-add
        merge_values = values.gather(2, merge_indices.view(bsz, 1, n_merge, 1).expand(-1, num_kv_heads, -1, head_dim))
        scale = (merge_mask / actual_budget.unsqueeze(1)).unsqueeze(-1)
        scale = torch.where(actual_budget.unsqueeze(1).unsqueeze(-1) == 0, torch.zeros_like(scale), scale)
        contributions = merge_values * scale
        contributions = contributions.unsqueeze(3).expand(-1, -1, -1, merge_budget, -1)
        contributions = contributions * valid_mask.view(bsz, 1, n_merge, merge_budget, 1)
        contributions = contributions.reshape(bsz, num_kv_heads, n_merge * merge_budget, head_dim)
        scatter_idx = target_positions.view(bsz, 1, n_merge * merge_budget, 1).expand(-1, num_kv_heads, -1, head_dim)

        values.scatter_add_(2, scatter_idx, contributions)
        return values

    def _triton_merge(self, values, merge_indices, kept_indices, attentions_per_kv, merge_budget):
        """Pre-computes cascading start targets and prefix sums, then merges in a single Triton kernel."""
        bsz, num_kv_heads, _, head_dim = values.shape
        n_merge = merge_indices.shape[1]
        n_kept = kept_indices.shape[1]
        attn_len = attentions_per_kv.shape[2]

        # 1. Pre-compute the cascading target
        base_idx_first = torch.searchsorted(kept_indices, merge_indices[:, 0:1], right=True)
        target_starts = torch.arange(n_merge, device=kept_indices.device).unsqueeze(0)
        target_starts += base_idx_first

        # 2. Prefix sum for O(1) suffix-mean in kernel: prefix_sum[i] = sum(attn[0:i])
        attn_prefix_sum = torch.nn.functional.pad(attentions_per_kv.cumsum(dim=-1), (1, 0))

        # 3. Pre-sampled random values for deterministic Bernoulli inside kernel
        rand_thresholds = torch.rand((bsz, num_kv_heads, n_merge), device=values.device)

        BLOCK_D = triton.next_power_of_2(head_dim)
        grid = (bsz, num_kv_heads, n_merge)

        _cam_merge_kernel[grid](
            value_states_ptr=values,
            merge_token_ids_ptr=merge_indices,
            kept_token_ids_ptr=kept_indices,
            merge_target_starts_ptr=target_starts,
            attn_cumsum_ptr=attn_prefix_sum,
            attn_weights_ptr=attentions_per_kv,
            rand_thresholds_ptr=rand_thresholds,
            num_kept_tokens=n_kept,
            merge_budget=merge_budget,
            seq_len=attn_len,
            v_stride_batch=values.stride(0),
            v_stride_head=values.stride(1),
            v_stride_seq=values.stride(2),
            v_stride_dim=values.stride(3),
            idx_stride_batch=merge_indices.stride(0),
            idx_stride_seq=merge_indices.stride(1),
            cs_stride_batch=attn_prefix_sum.stride(0),
            cs_stride_head=attn_prefix_sum.stride(1),
            cs_stride_seq=attn_prefix_sum.stride(2),
            attn_stride_batch=attentions_per_kv.stride(0),
            attn_stride_head=attentions_per_kv.stride(1),
            attn_stride_seq=attentions_per_kv.stride(2),
            head_dim=head_dim,
            BLOCK_D=BLOCK_D,
        )
        return values


if HAS_TRITON:

    @triton.jit
    def _cam_merge_kernel(
        value_states_ptr,
        merge_token_ids_ptr,
        kept_token_ids_ptr,
        merge_target_starts_ptr,
        attn_cumsum_ptr,
        attn_weights_ptr,
        rand_thresholds_ptr,
        num_kept_tokens,
        merge_budget,
        seq_len,
        v_stride_batch,
        v_stride_head,
        v_stride_seq,
        v_stride_dim,
        idx_stride_batch,
        idx_stride_seq,
        cs_stride_batch,
        cs_stride_head,
        cs_stride_seq,
        attn_stride_batch,
        attn_stride_head,
        attn_stride_seq,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        CaM scatter-add merge using cumulative attention with O(1) suffix-mean.
        Grid: (batch_size, num_kv_heads, n_merge)
        """
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        merge_idx = tl.program_id(2)

        # Load the merge token position and target window start
        merge_token_pos = tl.load(merge_token_ids_ptr + batch_idx * idx_stride_batch + merge_idx * idx_stride_seq)
        target_start = tl.load(merge_target_starts_ptr + batch_idx * idx_stride_batch + merge_idx * idx_stride_seq)

        # Calculate number of target tokens (handling edge near end of sequence)
        num_targets = tl.minimum(merge_budget, num_kept_tokens - target_start)
        if num_targets <= 0:
            return

        # Compute mean attention of suffix [target_start:] via cumulative sums (O(1))
        cumsum_base = attn_cumsum_ptr + batch_idx * cs_stride_batch + head_idx * cs_stride_head
        attn_suffix_sum = tl.load(cumsum_base + seq_len * cs_stride_seq) - tl.load(cumsum_base + target_start * cs_stride_seq)
        attn_suffix_len = seq_len - target_start
        mean_attn = attn_suffix_sum / attn_suffix_len

        # Load attention weight for the token being merged
        attn_weights_base = attn_weights_ptr + batch_idx * attn_stride_batch + head_idx * attn_stride_head
        merge_token_attn = tl.load(attn_weights_base + merge_token_pos * attn_stride_seq)

        # Calculate merge probability (with nan/inf safe-guards)
        if mean_attn == 0.0:
            merge_prob = 1.0 if merge_token_attn > 0 else 0.0
        else:
            merge_prob = merge_token_attn / mean_attn
        merge_prob = tl.minimum(tl.maximum(merge_prob, 0.0), 1.0)

        # Bernoulli draw using pre-computed random thresholds
        rand_val = tl.load(
            rand_thresholds_ptr + batch_idx * (tl.num_programs(1) * tl.num_programs(2)) + head_idx * tl.num_programs(2) + merge_idx
        )

        if merge_prob > rand_val:
            dim_offsets = tl.arange(0, BLOCK_D)
            dim_mask = dim_offsets < head_dim

            value_base = value_states_ptr + batch_idx * v_stride_batch + head_idx * v_stride_head

            # Load value vector of the token being merged
            merge_token_value = tl.load(value_base + merge_token_pos * v_stride_seq + dim_offsets * v_stride_dim, mask=dim_mask, other=0.0)
            contribution = merge_token_value / num_targets

            # Scatter-add contribution equally across target tokens
            for i in range(merge_budget):
                if i < num_targets:
                    target_offset = target_start + i
                    target_token_pos = tl.load(kept_token_ids_ptr + batch_idx * idx_stride_batch + target_offset * idx_stride_seq)

                    target_value_ptrs = value_base + target_token_pos * v_stride_seq + dim_offsets * v_stride_dim
                    tl.atomic_add(target_value_ptrs, contribution, mask=dim_mask)