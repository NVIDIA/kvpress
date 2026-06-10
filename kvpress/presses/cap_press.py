# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_query_states


@dataclass
class CapPress(ScorerPress):
    """
    This scoring module implements the CAPKV algorithm proposed in the paper
    (https://arxiv.org/abs/2604.25975).
    The scoring function is derived from the following formulation:

        A = I + sum_i w_i u_i u_i^T
        s_i = w_i * u_i^T A^{-1} u_i
        w_i = exp(tau * <k_i, mu_q>)

    where mu_q is a historical-query anchor and u_i is approximated by the
    value vector v_i.

    Note:
        1. For numerical stability, the keys and the historical-query anchor
           are L2-normalized before the alignment is computed (paper uses
           unnormalized dot products).
        2. A per-head max shift is applied before exponentiation to avoid
           numerical overflow.
        3. For RoPE-based models, historical query states are first rotated by
           an averaged future RoPE matrix before computing the query anchor.
           This is not described in the paper but is necessary to align the
           query and key representations in the RoPE-rotated space.
        4. The output-direction proxy is the raw value vector, u_i = v_i.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    tau : float, default=5.0
        Temperature parameter controlling the sharpness of the query-key
        alignment weights. Larger values produce more peaked distributions
        over tokens.
    n_future_positions : int, default=512
        Number of future positions used to compute the averaged RoPE rotation
        matrix for the query anchor.
    n_sink : int, default=4
        Number of initial tokens to exclude from compression (sink tokens).
    epsilon : float, default=1e-6
        Small constant for numerical stability in matrix inversion and
        weight computation.
    """

    compression_ratio: float = 0.0

    # hyperparameters.
    tau: float = 5.0

    # The following parameters are inherited from the implementation of Expected Attention (EA).
    n_future_positions: int = 512
    n_sink: int = 4
    epsilon: float = 1e-6

    def _query_states_pre_rope(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project hidden states to query states before applying averaged RoPE.

        Args:
            module: Attention module used to project hidden states to queries.
            hidden_states: Hidden states with shape [B, T, D_model].

        Returns:
            Query states with shape [B, H, T - n_sink, D_head].
        """
        hidden_states_no_sink = hidden_states[:, self.n_sink :]
        return get_prerope_query_states(module, hidden_states_no_sink)

    def _avg_rope_matrix(
        self,
        module: nn.Module,
        q_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Construct the average future RoPE rotation matrix.

        Args:
            module: Attention module exposing `rotary_emb` and `head_dim`.
            q_len: Current sequence length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Averaged RoPE rotation matrix with shape [D_head, D_head].
        """
        position_ids = (
            torch.arange(
                q_len,
                q_len + self.n_future_positions,
            )
            .unsqueeze(0)
            .to(device)
        )

        head_dim = module.head_dim
        dummy = torch.zeros((1, 1, head_dim), device=device, dtype=dtype)

        cos, sin = module.rotary_emb(dummy, position_ids)
        cos, sin = cos[0], sin[0]  # [n_future_positions, D_head]

        identity = torch.eye(head_dim, device=device, dtype=dtype)
        rotate_half = torch.zeros((head_dim, head_dim), device=device, dtype=dtype)
        half_dim = head_dim // 2

        rotate_half[half_dim:, :half_dim] = torch.eye(
            half_dim,
            device=device,
            dtype=dtype,
        )
        rotate_half[:half_dim, half_dim:] = -torch.eye(
            half_dim,
            device=device,
            dtype=dtype,
        )

        rotations = cos.unsqueeze(1) * identity + sin.unsqueeze(1) * rotate_half
        return rotations.mean(dim=0)

    def _apply_avg_rope(
        self,
        module: nn.Module,
        query_states: torch.Tensor,
        q_len: int,
    ) -> torch.Tensor:
        """
        Apply averaged future RoPE rotation to query states.

        Args:
            module: Attention module.
            query_states: Query states with shape [B, H, T', D_head].
            q_len: Current sequence length.

        Returns:
            RoPE-adjusted query states with shape [B, H, T', D_head].
        """
        rotation = self._avg_rope_matrix(
            module=module,
            q_len=q_len,
            device=query_states.device,
            dtype=query_states.dtype,
        )
        return torch.matmul(query_states, rotation.T)

    def _query_anchor(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the historical-query anchor.

        Args:
            module: Attention module.
            hidden_states: Hidden states with shape [B, T, D_model].

        Returns:
            Query anchor with shape [B, H, D_head].
        """
        q_len = hidden_states.shape[1]
        query_states = self._query_states_pre_rope(module, hidden_states)
        query_states = self._apply_avg_rope(module, query_states, q_len)
        return query_states.mean(dim=2)

    def _compute_query_key_statistic(
        self,
        query_anchor: torch.Tensor,
        keys_rep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the normalized query-key statistic.

        Args:
            query_anchor: Query anchor with shape [B, H, D_head].
            keys_rep: Keys repeated to attention-head space, with shape
                [B, H, T', D_head].

        Returns:
            Query-Key statistics with shape [B, H, T'].
        """
        query_anchor = F.normalize(query_anchor, p=2, dim=-1)
        keys_rep = F.normalize(keys_rep, p=2, dim=-1)

        statistic = torch.einsum("bhd,bhtd->bht", query_anchor, keys_rep)
        return statistic.clamp(-1.0, 1.0)

    def _compute_query_relevance_weights(
        self,
        statistic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert query-key statistics to non-negative relevance weights.

        Args:
            statistic: Query-key statistic with shape [B, H, T'].

        Returns:
            Relevance weights with shape [B, H, T'].
        """
        logits = self.tau * statistic
        logits = logits - logits.amax(dim=-1, keepdim=True)
        return torch.exp(logits)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute CAPKV retention scores.

        Args:
            module: Attention module.
            hidden_states: Hidden states with shape [B, T, D_model].
            keys: Cached keys with shape [B, H_kv, T_cache, D_head].
            values: Cached values with shape [B, H_kv, T_cache, D_head].
            attentions: Unused; included for KVPress scorer compatibility.
            kwargs: Unused; included for KVPress scorer compatibility.

        Returns:
            Retention scores with shape [B, H_kv, T_cache]. Larger values
            indicate higher retention priority.
        """
        del attentions, kwargs

        if keys.size(2) <= self.n_sink:
            raise ValueError(f"Input cache length ({keys.size(2)}) must be larger than " f"n_sink={self.n_sink}.")

        keys_no_sink = keys[:, :, self.n_sink :]
        values_no_sink = values[:, :, self.n_sink :]

        bsz, num_key_value_heads, q_len, d = keys_no_sink.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        # Repeat KV heads to attention-head space for query-aligned scoring.
        keys_rep = repeat_kv(keys_no_sink, num_key_value_groups)
        values_rep = repeat_kv(values_no_sink, num_key_value_groups)

        # Query relevance.
        query_anchor = self._query_anchor(module, hidden_states)
        statistic = self._compute_query_key_statistic(query_anchor, keys_rep)
        weights = self._compute_query_relevance_weights(statistic)

        # Capacity vectors: CAPKV uses raw values as output-direction proxies.
        capacity_vectors = values_rep

        sqrt_weights = torch.sqrt(weights + self.epsilon).unsqueeze(-1)
        scaled_vectors = capacity_vectors * sqrt_weights

        identity = torch.eye(
            d,
            device=scaled_vectors.device,
            dtype=scaled_vectors.dtype,
        ).view(1, 1, d, d)

        capacity_matrix = identity + torch.einsum(
            "bhtd,bhte->bhde",
            scaled_vectors,
            scaled_vectors,
        )

        capacity_matrix = capacity_matrix.to(dtype=torch.float32)

        capacity_vectors_t = capacity_vectors.transpose(2, 3).to(dtype=torch.float32)
        solution = torch.linalg.solve(capacity_matrix, capacity_vectors_t)

        leverage = (capacity_vectors_t * solution).sum(dim=2)
        scores = weights * leverage

        # Average repeated attention-head scores back into KV-head space.
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len)
        scores = scores.mean(dim=2)

        # If you want to fully reproduce the results from the paper,
        # you can remove the +1 and use only scores.max().item() as the score for sink.
        # standard top-k retention.
        scores = F.pad(scores, (self.n_sink, 0), value=scores.max().item() + 1)

        return scores
