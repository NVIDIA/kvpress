# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import QuantizedCache
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.adakv_press import AdaKVPress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values, get_prerope_query_states

logger = logging.getLogger(__name__)


@dataclass
class CAMPress(DecodingPress):
    """
    Cache Merging (CaM) KV cache compression during decoding.

    Instead of simply evicting low-importance tokens, CaM merges their value vectors
    into sequential neighbors before pruning. A Bernoulli merge mask, derived from the
    ratio of the evicted token's cumulative attention to the mean attention of its merge
    window, decides whether each merge occurs. This reduces the output
    perturbation caused by cache eviction.

    This implementation extends the original per-step algorithm to support batched
    eviction: tokens accumulate over ``compression_interval`` steps, then a bulk
    merge-and-prune pass is applied. Setting ``compression_interval=1`` creates
    the original per-step CaM behavior.

    Based on CaM (https://openreview.net/forum?id=LCTmppB165).

    Parameters
    ----------
    base_press : ScorerPress
        The scorer press used to compute importance scores for tokens.
    compression_interval : int, default=512
        Number of decoding steps between compression, i.e. compression will be applied
        every compression_interval steps.
    target_size : int, default=2048
        Target number of tokens to keep after compression.
    hidden_states_buffer_size : int, default=256
        Maximum number of hidden states to keep before compression. Larger values use
        more GPU memory. Note: Some presses don't need buffered hidden states and can
        set this to 0 to use only the current hidden state for compression scoring.
    merge_budget : int, default=32
        Number of sequential kept-token neighbors each evicted token's value is merged
        into. Smaller values concentrate the merged information; larger values spread it
        more evenly.
    """

    base_press: ScorerPress | AdaKVPress = None
    compression_interval: int = 512
    target_size: int = 2048
    hidden_states_buffer_size: int = 256
    merge_budget: int = 32

    def __post_init__(self):
        super().__post_init__()
        assert self.merge_budget > 0, "merge_budget must be positive "

        # To maintain cumulative attention sum across generation steps
        self._running_attn_sum: dict[int, torch.Tensor] = {}

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
        Merge evicted tokens' values into kept neighbors, then prune.

        Overrides `DecodingPress.compress` to implement the CaM merge-before-prune
        strategy instead of plain eviction.

        Args:
            module: The transformer attention module being compressed.
            hidden_states: Buffered hidden states from recent decoding steps
                (shape: [batch, buffer_len, hidden_dim]).
            keys: Key cache (shape: [batch, n_kv_heads, seq_len, head_dim]).
            values: Value cache (shape: [batch, n_kv_heads, seq_len, head_dim]).
            attentions: Cumulative attention scores summed over generation steps
                (shape: [batch, n_kv_heads, seq_len]).
            kwargs: Additional keyword arguments forwarded to the base press scorer.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Compressed (keys, values) with seq_len
            reduced to ``target_size``.

        Algorithm:
            1. **Score & select** — The base press scores every cached token. The
               ``n_to_evict`` lowest-scored tokens are marked for eviction; the top
               ``target_size`` are kept.
            2. **Pick merge candidates** — Among the evicted set, the ``k`` tokens with
               the highest scores (with ties broken by later sequence position) are
               selected for merging, where ``k = layer_step_counts[layer_idx]``
               (the number of new tokens since the last compression).
            3. **Cascading merge targets** — For each merge candidate, the
               ``merge_budget`` kept tokens immediately after it (in sequence order)
               form its merge window.
            4. **Merge probability** — The ratio of each merge token's cumulative
               attention to the mean cumulative attention of its window is computed
               ``clamp(A_i / avg(A_{j:j+m}), 0, 1)``.
            5. **Bernoulli sampling** — A binary merge mask is drawn from the
               probability above. Tokens that pass the mask have their value vectors
               divided by the window size and scatter-added into the window targets.
            6. **Physical pruning** — Evicted key/value entries are removed from the
               cache, and the cumulative attention buffer is pruned to match.
        """

        layer_idx = int(module.layer_idx)
        cache_len = keys.shape[2]

        n_to_evict = cache_len - self.target_size

        target_compression_ratio = self._find_target_compression_ratio(cache_len, self.target_size)

        if n_to_evict <= 0:
            return keys, values

        # Temporary override base press ratio to get correct topK scores
        old_cr = self.base_press.compression_ratio
        self.base_press.compression_ratio = target_compression_ratio
        scores = self.base_press.score(module, hidden_states, keys, values, None, kwargs)
        self.base_press.compression_ratio = old_cr

        bsz, num_key_value_heads, seq_len, head_dim = keys.shape

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

        n_to_merge = merge_indices.shape[1]

        base_idx_first = torch.searchsorted(kept_indices, merge_indices[:, 0:1], right=True)
        target_starts = torch.arange(n_to_merge, device=kept_indices.device).unsqueeze(0) + base_idx_first

        # 2. Build target window indices: [bsz, n_to_merge, merge_budget]
        offsets = torch.arange(self.merge_budget, device=kept_indices.device)
        window_idx = target_starts.unsqueeze(-1) + offsets.view(1, 1, -1)
        valid_mask = window_idx < self.target_size
        window_idx = window_idx.clamp(max=self.target_size - 1)
        target_positions = kept_indices.gather(1, window_idx.view(bsz, -1)).view(bsz, n_to_merge, self.merge_budget)

        # 3. Actual budget per merge token
        actual_budget = valid_mask.sum(dim=-1)

        # 4. Window mean via cumsum (from target_start to min(target_start + merge_budget, seq_len))
        attn_cumsum = torch.nn.functional.pad(attentions.cumsum(dim=-1), (1, 0))
        start_sum = attn_cumsum.gather(2, target_starts.unsqueeze(1).expand(-1, num_key_value_heads, -1))
        window_end = (target_starts + self.merge_budget).clamp(max=seq_len)
        end_sum = attn_cumsum.gather(2, window_end.unsqueeze(1).expand(-1, num_key_value_heads, -1))
        window_sum = end_sum - start_sum
        window_len = (window_end - target_starts).unsqueeze(1)
        mean_attn = window_sum / window_len

        # 5. Merge probability
        merge_token_attn = attentions.gather(2, merge_indices.unsqueeze(1).expand(-1, num_key_value_heads, -1))
        merge_prob = merge_token_attn / mean_attn
        merge_prob = torch.where(torch.isnan(merge_prob), torch.zeros_like(merge_prob), merge_prob)
        merge_prob = torch.where(torch.isinf(merge_prob), torch.ones_like(merge_prob), merge_prob)
        merge_prob = merge_prob.clamp(0, 1)

        # 6. Bernoulli sampling
        merge_mask = torch.bernoulli(merge_prob)

        # 7. Build contributions and scatter-add
        merge_values = values.gather(
            2, merge_indices.view(bsz, 1, n_to_merge, 1).expand(-1, num_key_value_heads, -1, head_dim)
        )
        scale = (merge_mask / actual_budget.unsqueeze(1)).unsqueeze(-1)
        scale = torch.where(actual_budget.unsqueeze(1).unsqueeze(-1) == 0, torch.zeros_like(scale), scale)
        contributions = merge_values * scale
        contributions = contributions.unsqueeze(3).expand(-1, -1, -1, self.merge_budget, -1)
        contributions = contributions * valid_mask.view(bsz, 1, n_to_merge, self.merge_budget, 1)
        contributions = contributions.reshape(bsz, num_key_value_heads, n_to_merge * self.merge_budget, head_dim)
        scatter_idx = target_positions.view(bsz, 1, n_to_merge * self.merge_budget, 1).expand(
            -1, num_key_value_heads, -1, head_dim
        )

        values.scatter_add_(2, scatter_idx, contributions)

        # Physical Pruning
        kept_indices_expand = kept_indices.view(bsz, 1, self.target_size, 1).expand(
            bsz, num_key_value_heads, self.target_size, head_dim
        )
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
        """
        Forward hook that manages cumulative attention tracking and interval-based compression.

        Extends `DecodingPress.forward_hook` with per-step attention accumulation.

        This hook:
        1. Detects when we're in decoding phase (not prefilling)
        2. Accumulates hidden states in a buffer
        3. Accumulates cumulative attention A_bar = sum(A^k) in a buffer
        4. Applies compression every N steps
        5. Clears the buffer after compression
        """
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        layer_idx = int(module.layer_idx)

        # Only operate during decoding
        if kwargs["cache_position"][-1] <= q_len:
            return output

        # All hidden_states_buffer code is borrowed from DecodingPress
        self.hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())

        cache_layer = cache.layers[module.layer_idx]
        keys, values = extract_keys_and_values(cache, layer_idx)
        bsz, num_key_value_heads, seq_len, _ = keys.shape

        # Accumulate Cumulative Attention over generation steps
        attentions = output[1] if len(output) > 1 and output[1] is not None else None
        if attentions is None:
            attentions = self._compute_current_token_attention(module, hidden_states, keys, kwargs)
        else:
            attentions = attentions[:, :, -1:, :]

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
        if (self.layer_step_counts[layer_idx] >= self.compression_interval and seq_len > self.target_size) or (
            q_len >= self.target_size
        ):

            # Apply compression using cumulative attention scores and buffered hidden states
            attn_squeezed = self._running_attn_sum[layer_idx]
            buffered_hidden_states = torch.cat(self.hidden_states_buffer[layer_idx], dim=1)
            keys, values = self.compress(module, buffered_hidden_states, keys, values, attn_squeezed, kwargs)

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
            # Always clear the buffer after compression - otherwise there's a mismatch between
            # hidden states buffer and kv cache
            self.hidden_states_buffer[layer_idx] = []

        self.hidden_states_buffer[layer_idx] = (
            self.hidden_states_buffer[layer_idx][-self.hidden_states_buffer_size :]
            if self.hidden_states_buffer_size > 0
            else []
        )

        return output

    def reset(self):
        """Reset per-sequence state."""
        super().reset()
        self._running_attn_sum = {}

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
