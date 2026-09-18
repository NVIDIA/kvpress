# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import compute_n_kept, get_prerope_query_states


@dataclass
class DropKVPress(ScorerPress):
    """
    DropKV: KV cache eviction via Decoupled Residual-Output Perturbation.

    Choosing which KV pairs to evict so as to minimize the resulting attention
    output perturbation is an NP-hard combinatorial problem, since the pairs
    interact through the softmax normalization. DropKV decouples that joint
    decision into independent per-token scores: each KV pair is scored by the
    residual-output perturbation it alone would cause, which sidesteps the
    combinatorial intractability and admits a constant-factor approximation
    guarantee under the cone condition studied in the paper.

    Evicting key-value pair j renormalizes the softmax over the remaining
    entries, so for query i the residual has the closed form

        o_i^(-j) - o_i = p_ij / (1 - p_ij) * (o_i - v_j)

    where p_ij is the attention probability of query i on position j and o_i is
    the current attention output. DropKV accumulates the squared residual over
    the most recent `window_size` queries,

        score_j = sum_i ( p_ij / (1 - p_ij) )^2 * || v_j - o_i ||^2

    so the score is exact for a single eviction rather than a proxy such as the
    raw attention weight. Scores are averaged over the query heads sharing a KV
    head, smoothed along the sequence with average pooling over `kernel_size`
    positions, and the most recent `window_size` positions are always retained.

    Based on DropKV (https://openreview.net/forum?id=MqfNzH3TVH).

    Note: The paper additionally provides fused Triton kernels that compute the
    scores without materializing the attention matrix. This implementation is
    plain PyTorch and follows kvpress's readability-first convention, so it does
    not reproduce the reported scoring-kernel speedup.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    window_size : int, default=32
        Number of recent queries used for scoring and recent KV pairs that are
        always retained.
    kernel_size : int, default=7
        Odd-sized average-pooling kernel used to smooth token scores.
    epsilon : float, default=1e-6
        Numerical-stability constant used in the sensitivity weight.
    """

    compression_ratio: float = 0.0
    window_size: int = 32
    kernel_size: int = 7
    epsilon: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {self.kernel_size}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

    def _get_window_queries(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Project and rotate the most recent query window."""
        if hidden_states.shape[1] < self.window_size:
            raise ValueError(f"Query length {hidden_states.shape[1]} must be at least window_size={self.window_size}")

        query_states = get_prerope_query_states(module, hidden_states[:, -self.window_size :])
        cos, sin = position_embeddings
        cos = cos[:, -self.window_size :]
        sin = sin[:, -self.window_size :]
        return (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

    def _compute_scores(
        self,
        query_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DropKV scores in query-head space, then average GQA groups."""
        if keys.shape != values.shape:
            raise ValueError(f"keys and values must have the same shape, got {keys.shape} and {values.shape}")

        bsz, num_kv_heads, seq_len, head_dim = keys.shape
        num_query_heads = query_states.shape[1]
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(f"Query heads {num_query_heads} must be divisible by KV heads {num_kv_heads}")

        num_groups = num_query_heads // num_kv_heads
        query_len = query_states.shape[2]
        if query_len != self.window_size:
            raise ValueError(f"Expected {self.window_size} query states, got {query_len}")
        if seq_len < query_len:
            raise ValueError(f"KV length {seq_len} must be at least query length {query_len}")

        repeated_keys = repeat_kv(keys, num_groups)
        repeated_values = repeat_kv(values, num_groups)

        attention_weights = torch.matmul(query_states, repeated_keys.transpose(2, 3)) / math.sqrt(head_dim)

        causal_mask = torch.full(
            (query_len, seq_len),
            float("-inf"),
            device=attention_weights.device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=seq_len - query_len + 1)
        attention_weights += causal_mask
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attention_output = torch.matmul(attention_weights, repeated_values)

        # Preserve the original precision path: probabilities are rounded to
        # the query dtype before all sensitivity terms are evaluated in fp32.
        attention_float = attention_weights.float()
        sensitivity_weights = (attention_float / (1.0 - attention_float + self.epsilon)).square()

        weight_sum = sensitivity_weights.sum(dim=-2)
        value_norm_squared = repeated_values.float().square().sum(dim=-1)
        term_a = weight_sum * value_norm_squared

        weighted_outputs = torch.matmul(sensitivity_weights.transpose(-1, -2), attention_output.float())
        term_b = 2.0 * torch.sum(repeated_values.float() * weighted_outputs, dim=-1)

        output_norm_squared = attention_output.float().square().sum(dim=-1)
        term_c = torch.matmul(sensitivity_weights.transpose(-1, -2), output_norm_squared.unsqueeze(-1)).squeeze(-1)

        scores = term_a - term_b + term_c
        scores = scores.view(bsz, num_kv_heads, num_groups, seq_len)
        return scores.mean(dim=2)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        del attentions

        query_states = self._get_window_queries(module, hidden_states, kwargs["position_embeddings"])
        scores = self._compute_scores(query_states, keys, values)

        scores = F.avg_pool1d(
            scores,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        # Recent positions are both the observation window and the context most
        # immediately relevant to decoding, so guarantee that top-k retains them.
        protected_score = scores.amax(dim=-1, keepdim=True) + 1.0
        scores[:, :, -self.window_size :] = protected_score
        return scores

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0 or keys.shape[2] < self.window_size:
            return keys, values

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        seq_len = keys.shape[2]
        n_kept = min(seq_len, max(compute_n_kept(seq_len, self.compression_ratio), self.window_size))
        indices = scores.topk(n_kept, dim=-1, largest=True).indices
        indices = indices.sort(dim=-1).values
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])

        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values
