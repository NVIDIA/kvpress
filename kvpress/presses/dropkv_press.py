# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_query_states


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
    positions, and the observation window is always retained. Under the eager
    attention implementation the probabilities are taken from the forward pass
    instead of being recomputed.

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
        always retained. Clamped to the query and cache lengths available in the
        current forward pass, so decoding presses can score short buffers.
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

    def _effective_window_size(self, q_len: int, k_len: int) -> int:
        """Observation window, clamped to what the current forward pass provides.

        Decoding presses call the scorer with only a handful of buffered queries,
        so the nominal ``window_size`` is not always available.
        """
        return max(1, min(self.window_size, q_len, k_len))

    def _get_window_queries(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        window_size: int,
    ) -> torch.Tensor:
        """Project and rotate the most recent query window."""
        cos, sin = position_embeddings
        return get_query_states(
            module,
            hidden_states[:, -window_size:],
            (cos[:, -window_size:], sin[:, -window_size:]),
        )

    def _compute_window_probabilities(self, query_states: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Attention probabilities of the window queries over the whole cache.

        Only used when the attention weights are not already available from the
        forward pass, i.e. outside of the eager attention implementation.
        """
        num_query_heads, query_len = query_states.shape[1], query_states.shape[2]
        num_kv_heads, seq_len, head_dim = keys.shape[1], keys.shape[2], keys.shape[3]
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(f"Query heads {num_query_heads} must be divisible by KV heads {num_kv_heads}")

        repeated_keys = repeat_kv(keys, num_query_heads // num_kv_heads)
        attention_weights = torch.matmul(query_states, repeated_keys.transpose(2, 3)) / math.sqrt(head_dim)

        causal_mask = torch.full(
            (query_len, seq_len),
            float("-inf"),
            device=attention_weights.device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=seq_len - query_len + 1)
        attention_weights += causal_mask
        return F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    def _compute_scores(
        self,
        probabilities: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate the squared residual over the query window, then average GQA groups."""
        if keys.shape != values.shape:
            raise ValueError(f"keys and values must have the same shape, got {keys.shape} and {values.shape}")

        bsz, num_kv_heads, seq_len, _ = keys.shape
        num_query_heads, query_len = probabilities.shape[1], probabilities.shape[2]
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(f"Query heads {num_query_heads} must be divisible by KV heads {num_kv_heads}")
        if seq_len < query_len:
            raise ValueError(f"KV length {seq_len} must be at least query length {query_len}")
        if probabilities.shape[-1] != seq_len:
            raise ValueError(f"Expected probabilities over {seq_len} positions, got {probabilities.shape[-1]}")

        num_groups = num_query_heads // num_kv_heads
        repeated_values = repeat_kv(values, num_groups)

        attention_output = torch.matmul(probabilities, repeated_values)

        # Preserve the original precision path: probabilities are rounded to
        # the query dtype before all sensitivity terms are evaluated in fp32.
        attention_float = probabilities.float()
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
        window_size = self._effective_window_size(hidden_states.shape[1], keys.shape[2])

        if attentions is not None:
            # Eager attention already materialized the probabilities. Unlike SnapKV,
            # DropKV needs every column: the attention output is recomputed from them.
            probabilities = attentions[..., -window_size:, :]
        else:
            query_states = self._get_window_queries(module, hidden_states, kwargs["position_embeddings"], window_size)
            probabilities = self._compute_window_probabilities(query_states, keys)

        scores = self._compute_scores(probabilities, keys, values)

        scores = F.avg_pool1d(
            scores,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        # Recent positions are both the observation window and the context most
        # immediately relevant to decoding, so guarantee that top-k retains them.
        protected_score = scores.amax(dim=-1, keepdim=True) + 1.0
        scores[:, :, -window_size:] = protected_score
        return scores
