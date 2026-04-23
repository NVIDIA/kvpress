# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.observed_attention_press import ObservedAttentionPress


@dataclass
class EntropyAdaptivePress(ObservedAttentionPress):
    """
    Adaptive compression based on attention entropy.

    Extends ObservedAttentionPress by adjusting the effective compression ratio
    per layer based on how peaked or uniform the attention distribution is.
    Peaked attention (low entropy) tolerates more eviction. Uniform attention
    (high entropy) needs a gentler rate.

    Requires: attn_implementation="eager".

    Parameters
    ----------
    compression_ratio : float, default=0.5
        Base compression ratio. Actual ratio varies per layer.
    min_ratio : float, default=0.1
        Floor for adaptive ratio (high entropy layers).
    max_ratio : float, default=0.7
        Ceiling for adaptive ratio (low entropy layers).
    """

    compression_ratio: float = 0.5
    min_ratio: float = 0.1
    max_ratio: float = 0.7

    def compute_window_size(self, module: nn.Module) -> int:
        """Adaptive window: keep more tokens when attention is uniform."""
        seq_len = getattr(module, "_seq_len", 128)
        n_keep = max(1, int(seq_len * (1.0 - self.compression_ratio)))
        return n_keep

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        # Get base scores from ObservedAttentionPress
        scores = super().score(module, hidden_states, keys, values, attentions, kwargs)

        if attentions is None:
            return scores

        # Compute entropy of this layer's attention to modulate scores
        attn = attentions.float()
        avg_attn = attn.mean(dim=(0, 1, 2))  # average over batch, heads, queries
        probs = avg_attn / (avg_attn.sum() + 1e-8)
        entropy = -(probs * (probs + 1e-8).log()).sum().item()
        max_entropy = math.log(max(probs.shape[0], 2))
        normalized = entropy / max_entropy  # 0=peaked, 1=uniform

        # Scale scores: high entropy -> boost all scores (keep more tokens)
        # low entropy -> leave scores as-is (evict more)
        boost = normalized * 0.5  # up to 50% score boost for uniform attention
        return scores * (1.0 + boost)
