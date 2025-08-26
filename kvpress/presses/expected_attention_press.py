# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.utils import get_query_states


@dataclass
class ExpectedAttentionPress(ScorerPress):
    """
    Expected attention-based KV cache compression.

    Computes importance scores based on expected attention that future queries
    will pay to current key-value pairs. Uses statistical modeling of query
    patterns and RoPE rotation matrices to predict future attention.
    In particular:
        1. Compute the mean and covariance matrix of the queries before RoPE.
        2. Compute the RoPE rotation matrix R on next n_future_positions and average it
        3. Apply R to the mean and covariance matrice of the queries.
        4. As attention A = exp(Q @ K / sqrt(d)), we compute the expected attention
        E(A) = exp(K @ mean.T / sqrt(d) + 1/2 K @ cov @ K.T / d)
        5. Rescale the scores using (scores + epsilon) * ||V||_2

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    n_future_positions : int, default=512
        Number of future positions to consider when computing expected attention.
    n_sink : int, default=4
        Number of initial tokens to exclude from compression (sink tokens).
        Preserves first few tokens due to "sink attention" phenomenon where models
        assign high attention to early tokens regardless of semantic importance.
    use_covariance : bool, default=True
        Whether to include covariance information in expected attention computation.
        When True, uses both mean and covariance of query distributions for more
        accurate but computationally expensive scoring. When False, uses only mean.
    use_vnorm : bool, default=True
        Whether to rescale scores using value vector norms.
        Rescales expected attention scores by L2 norm of corresponding value vectors:
        (scores + epsilon) * ||V||₂. Accounts for magnitude of attended information.
    epsilon : float, default=0.0
        Small constant added to scores before value norm rescaling for numerical stability.
    """

    compression_ratio: float = 0.0
    n_future_positions: int = 512
    n_sink: int = 4
    use_covariance: bool = True
    use_vnorm: bool = True
    epsilon: float = 0.0

    def get_query_statistics(self, module: nn.Module, hidden_states: torch.Tensor):
        """
        Compute the mean and covariance matrix of the queries
        """

        q_len = hidden_states.shape[1]
        head_dim = module.head_dim

        # Remove first hidden_states that likely contain outliers
        h = hidden_states[:, self.n_sink :]
        query_states = get_query_states(module, h)

        # Query mean
        mu = query_states.mean(dim=2, keepdim=True)

        # Query covariance
        cov = None
        if self.use_covariance:
            centered_states = query_states - mu
            cov = torch.einsum("bnsi,bnsj->bnij", centered_states, centered_states) / h.shape[1]
        mu = mu.squeeze(2)

        # RoPE rotation matrix on next n_future_positions
        position_ids = torch.arange(q_len, q_len + self.n_future_positions).unsqueeze(0).to(mu.device)
        cos, sin = module.rotary_emb(mu, position_ids)
        cos, sin = cos[0], sin[0]

        Id = torch.eye(head_dim, device=cos.device, dtype=cos.dtype)
        P = torch.zeros((head_dim, head_dim), device=cos.device, dtype=cos.dtype)
        P[head_dim // 2 :, : head_dim // 2], P[: head_dim // 2, head_dim // 2 :] = torch.eye(head_dim // 2), -torch.eye(
            head_dim // 2
        )
        R = cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P

        # Apply average rotation to the mean and covariance
        R = R.mean(dim=0).to(mu.device)
        mu = torch.matmul(mu, R.T)
        if self.use_covariance:
            cov = torch.matmul(R, torch.matmul(cov, R.T))

        # Instead of using the average rotation matrix, we could use a mixture of gaussian statistics to
        # estimate mean and covariance. Estimation is better, but end-to-end performance was lower.
        # mu = torch.einsum("bhj, fij -> bhfi", mu, R)
        # mean_mu = mu.mean(dim=2, keepdim=True)
        # if self.use_covariance:
        #     cov = torch.einsum("fki, bhkl, fjl -> bhfij", R, cov, R)
        #     cov = cov.mean(dim=2)
        #     cov += torch.einsum("bhfi, bhfj -> bhji", mu - mean_mu, mu - mean_mu) / self.n_future_positions
        # mu = mean_mu.squeeze(2)

        return mu, cov

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        # Remove sink tokens
        assert keys.size(2) > self.n_sink, f"Input should contain more tokens than n_sink={self.n_sink}"
        keys = keys[:, :, self.n_sink :]
        values = values[:, :, self.n_sink :]

        # Compute query statistics
        mean_query, cov_query = self.get_query_statistics(module, hidden_states)

        # Compute scores
        bsz, num_key_value_heads, q_len, d = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        keys = repeat_kv(keys, num_key_value_groups).transpose(2, 3)
        scores = torch.matmul(mean_query.unsqueeze(2), keys).squeeze(2) / math.sqrt(d)
        if self.use_covariance:
            scores += torch.einsum("bhin, bhij, bhjn->bhn", keys, cov_query, keys) / d / 2
        scores = F.softmax(scores, dim=-1)

        # Average scores across groups
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len)
        scores = scores.mean(dim=2)

        # Rescale scores by the norm of the values
        if self.use_vnorm:
            scores = (scores + self.epsilon) * values.norm(dim=-1)

        # Add back the sink tokens. Use max score to make sure they are not pruned.
        scores = F.pad(scores, (self.n_sink, 0), value=scores.max().item())

        return scores
