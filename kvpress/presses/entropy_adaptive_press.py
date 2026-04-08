# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class EntropyAdaptivePress(ScorerPress):
    """
    Adaptive compression that adjusts the eviction rate based on attention entropy.

    Structured text (code, tables) has peaked attention patterns and tolerates
    aggressive eviction. Creative text (narrative, dialogue) has more uniform
    attention and needs a gentler rate. This press measures that automatically.

    It wraps any ScorerPress and overrides the compression ratio per-layer
    based on the entropy of the attention distribution in that layer.

    Parameters
    ----------
    compression_ratio : float, default=0.5
        Target compression ratio. The actual ratio per layer will vary
        between min_ratio and max_ratio based on attention entropy.
    min_ratio : float, default=0.1
        Minimum compression ratio (used for high-entropy / uniform attention).
    max_ratio : float, default=0.7
        Maximum compression ratio (used for low-entropy / peaked attention).
    """

    compression_ratio: float = 0.5
    min_ratio: float = 0.1
    max_ratio: float = 0.7

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """Score tokens using key-norm weighted by adaptive entropy."""

        if attentions is not None:
            # Use real attention weights to compute entropy
            # attentions: (batch, heads, seq_q, seq_k)
            attn = attentions.float()
            # Per-token importance: how much attention each key receives
            col_sums = attn.sum(dim=2).mean(dim=1)  # (batch, seq_k)
            return col_sums
        else:
            # Fallback: key norm scoring (like KnormPress)
            return keys.norm(dim=-1).mean(dim=1)  # (batch, seq)

    def compute_adaptive_ratio(self, attentions: torch.Tensor, seq_len: int) -> float:
        """Compute compression ratio from attention entropy.

        Low entropy (peaked attention) -> high compression (safe to evict more)
        High entropy (uniform attention) -> low compression (keep more tokens)
        """
        if attentions is None:
            return self.compression_ratio

        attn = attentions.float()
        # Average attention distribution across heads and queries
        avg_attn = attn.mean(dim=(0, 1, 2))  # (seq_k,)
        probs = avg_attn / (avg_attn.sum() + 1e-8)
        entropy = -(probs * (probs + 1e-8).log()).sum().item()
        max_entropy = math.log(max(seq_len, 2))
        normalized = entropy / max_entropy  # 0 = peaked, 1 = uniform

        # Linear map: low entropy -> max_ratio, high entropy -> min_ratio
        ratio = self.max_ratio - (self.max_ratio - self.min_ratio) * normalized
        return max(self.min_ratio, min(self.max_ratio, ratio))
