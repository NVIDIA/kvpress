# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class StreamingLLMPress(ScorerPress):
    """
    StreamingLLM: Window-based KV cache compression with sink tokens.

    Implements sliding window approach preserving first few tokens (sink tokens)
    and most recent tokens, while pruning middle tokens.

    Based on StreamingLLM (https://arxiv.org/abs/2309.17453).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    n_sink : int, default=4
        Number of initial tokens to always preserve (sink tokens).
        These tokens are never pruned and serve as "attention sinks" that help
        maintain model stability. Language models often assign high attention
        weights to early tokens regardless of semantic content.
    """

    compression_ratio: float = 0.0
    n_sink: int = 4

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        q_len = hidden_states.shape[1]
        assert q_len > self.n_sink, f"Input should contain more tokens than n_sink={self.n_sink}"
        n_pruned = q_len - int(q_len * (1 - self.compression_ratio))
        scores = torch.ones_like(keys[..., 0])
        scores[:, :, self.n_sink : self.n_sink + n_pruned] = 0

        return scores
