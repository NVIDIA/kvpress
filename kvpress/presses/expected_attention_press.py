# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class ExpectedAttentionPress(ScorerPress):
    """
    Compute scores based on the expected attention on next positions. To do so
        1. Compute the mean and covariance matrix of the queries before RoPE.
        2. Compute the RoPE rotation matrix R on next n_future_positions and average it
        3. Apply R to the mean and covariance matrice of the queries.
        4. As attention A = exp(Q @ K / sqrt(d)), we compute the expected attention
        E(A) = exp(K @ mean.T / sqrt(d) + 1/2 K @ cov @ K.T / d)
        5. Rescale the scores by the norm of the values
    The first n_sink tokens are removed from calculations (sink attention phenomenon).
    """

    compression_ratio: float = 0.0
    n_future_positions: int = 512
    n_sink: int = 4
    use_covariance: bool = True
    use_vnorm: bool = True

    def get_query_statistics(self, module: nn.Module, hidden_states: torch.Tensor):
        """
        Compute the mean and covariance matrix of the queries
        """

        bsz, q_len, _ = hidden_states.shape
        n, d = module.config.num_attention_heads, module.head_dim

        # Remove first hidden_states that likely contain outliers
        h = hidden_states[:, self.n_sink :]

        if hasattr(module, "q_proj"):
            Wq = module.q_proj.weight
        elif hasattr(module, "qkv_proj"):
            Wq = module.qkv_proj.weight[: n * d]
        else:
            raise NotImplementedError(f"ExpectedAttentionPress not yet implemented for {module.__class__}.")

        # Query mean
        mean_h = torch.mean(h, dim=1, keepdim=True)
        mu = torch.matmul(mean_h, Wq.T).squeeze(1)
        mu = mu.view(bsz, n, d)

        # Query covariance
        cov = None
        if self.use_covariance:
            h = h - mean_h
            cov = torch.matmul(h.transpose(1, 2), h) / h.shape[1]
            cov = torch.matmul(Wq, torch.matmul(cov, Wq.T))  # TODO: not optimal
            cov = cov.view(bsz, n, d, n, d).diagonal(dim1=1, dim2=3)
            cov = cov.permute(0, 3, 1, 2)

        # RoPE rotation matrix on next n_future_positions
        position_ids = torch.arange(q_len, q_len + self.n_future_positions).unsqueeze(0).to(mu.device)
        cos, sin = module.rotary_emb(mu, position_ids)
        cos, sin = cos[0], sin[0]

        Id = torch.eye(d, device=cos.device, dtype=cos.dtype)
        P = torch.zeros((d, d), device=cos.device, dtype=cos.dtype)
        P[d // 2 :, : d // 2], P[: d // 2, d // 2 :] = torch.eye(d // 2), -torch.eye(d // 2)
        R = cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P

        # Apply average rotation to the mean and covariance
        R = R.mean(dim=0)
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
            scores = scores * values.norm(dim=-1)

        # Add back the sink tokens. Use max score to make sure they are not pruned.
        scores = F.pad(scores, (self.n_sink, 0), value=scores.max().item())

        return scores
