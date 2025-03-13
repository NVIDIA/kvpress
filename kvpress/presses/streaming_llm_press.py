# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class StreamingLLMPress(ScorerPress):
    """
    Prune a fixed number of KV pairs at the beginning and end of the sequence (https://arxiv.org/abs/2309.17453)
    We keep the first n_sink tokens and the last n_local tokens.
    n_local is computed using the compression ratio.

    Note that the original implementation https://github.com/mit-han-lab/streaming-llm additionally rerotates keys.
    This can be achieved by using
    press = KeyRerotationPress(press=StreamingLLMPress(compression_ratio, n_sink))
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
