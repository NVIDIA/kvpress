# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.phi3.modeling_phi3 import Phi3Attention

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class ExpectedAttentionPress(ScorerPress):
    """
    Expected attention-based KV cache compression.
    
    This method computes importance scores based on the expected attention that
    future query tokens will pay to current key-value pairs. It uses statistical
    modeling of query patterns to predict which tokens will be most important
    for upcoming attention computations.
    
    The algorithm works by:
    1. Computing mean and covariance statistics of queries before RoPE application
    2. Modeling future query positions using RoPE rotation matrices
    3. Computing expected attention weights: E[A] = exp(K @ μ^T / √d + ½K @ Σ @ K^T / d)
    4. Optionally rescaling scores using value norms: (scores + ε) * ||V||₂
    5. Excluding sink tokens from calculations to handle sink attention phenomenon
    
    This approach is particularly effective because:
    - It predicts future attention patterns rather than relying on past patterns
    - It accounts for positional encoding effects through RoPE modeling
    - It can incorporate both mean and variance information from query distributions
    - It handles the sink attention phenomenon by excluding initial tokens
    """

    compression_ratio: float = 0.0
    """
    Fraction of key-value pairs to remove during compression.
    See ScorerPress.compression_ratio for detailed description.
    """
    
    n_future_positions: int = 512
    """
    Number of future positions to consider when computing expected attention.
    
    This parameter controls how far ahead the method looks when predicting
    future attention patterns. The RoPE rotation matrices are computed for
    this many future positions and averaged to estimate expected attention.
    
    Larger values:
    - Consider longer-range future dependencies
    - May provide more stable attention estimates
    - Require more computation for RoPE matrix calculations
    
    Smaller values:
    - Focus on near-term attention patterns
    - Are more computationally efficient
    - May miss longer-range dependencies
    
    Default of 512 provides a good balance for most applications.
    """
    
    n_sink: int = 4
    """
    Number of initial tokens to exclude from compression (sink tokens).
    
    The "sink attention" phenomenon refers to the tendency of language models
    to assign high attention weights to the first few tokens in a sequence,
    regardless of their semantic importance. These tokens act as "attention sinks."
    
    This parameter specifies how many initial tokens to always preserve:
    - 0: No sink tokens (may hurt performance)
    - 4: Preserve first 4 tokens (default, works well for most models)
    - 8+: More conservative, preserves more initial tokens
    
    Preserving sink tokens typically improves model performance after compression.
    """
    
    use_covariance: bool = True
    """
    Whether to include covariance information in expected attention computation.
    
    When True, the method computes both mean and covariance of query distributions
    and uses both in the expected attention formula. When False, only the mean
    is used, which is computationally cheaper but may be less accurate.
    
    - True: Use full statistical model (mean + covariance) - more accurate
    - False: Use only mean statistics - faster but potentially less precise
    
    The covariance term captures the uncertainty in query patterns and generally
    improves compression quality at the cost of additional computation.
    """
    
    use_vnorm: bool = True
    """
    Whether to rescale scores using value vector norms.
    
    When True, the computed expected attention scores are rescaled by the L2 norm
    of the corresponding value vectors: (scores + epsilon) * ||V||₂. This helps
    account for the magnitude of values when determining token importance.
    
    - True: Rescale by value norms (generally recommended)
    - False: Use raw expected attention scores
    
    Value norm rescaling typically improves compression quality by considering
    both attention patterns and the magnitude of information being attended to.
    """
    
    epsilon: float = 0.0
    """
    Small constant added to scores before value norm rescaling.
    
    This parameter is only used when use_vnorm=True. It's added to the expected
    attention scores before multiplying by value norms to provide numerical
    stability and prevent issues with very small attention scores.
    
    - 0.0: No additional constant (default)
    - Small positive value: Provides numerical stability
    
    Usually the default of 0.0 works well, but small positive values can help
    in cases where numerical instability is observed.
    """

    def get_query_statistics(self, module: nn.Module, hidden_states: torch.Tensor):
        """
        Compute the mean and covariance matrix of the queries
        """

        bsz, q_len, _ = hidden_states.shape
        n, d = module.config.num_attention_heads, module.head_dim

        # Remove first hidden_states that likely contain outliers
        h = hidden_states[:, self.n_sink :]

        if isinstance(module, (Qwen3Attention, Gemma3Attention)):
            # Qwen and Gemma use QK norm, which is not compatible with ExpectedAttentionPress (for now)
            raise NotImplementedError(f"ExpectedAttentionPress not yet implemented for {module.__class__}.")
        elif isinstance(module, Phi3Attention):
            Wq = module.qkv_proj.weight[: n * d]
        elif hasattr(module, "q_proj"):
            # Assume Llama-like attention layer
            Wq = module.q_proj.weight  # type: ignore[assignment]
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
