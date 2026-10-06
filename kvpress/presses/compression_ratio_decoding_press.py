# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from kvpress.presses.base_press import is_prefilling
from kvpress.presses.decoding_press import DecodingPress


@dataclass
class CompressionRatioDecodingPress(DecodingPress):
    """
    A decoding press that keeps a fixed fraction of all tokens seen so far.

    Unlike `DecodingPress`, which compresses to a fixed absolute `target_size`,
    this subclass derives the target size from the number of tokens seen so far.

    Parameters
    ----------
    base_press : ScorerPress
        The scorer press used to compute importance scores for tokens.
    target_compression_ratio : float
        Fraction of all tokens seen so far to remove during decoding.
    compression_interval : int, default=512
        Number of decoding steps between compression.
    hidden_states_buffer_size : int, default=256
        Maximum number of hidden states to keep before compression.

    Notes
    -----
    The press counts the tokens of the prefill and of every decoding step. If the prefill ran outside of the press
    context (as in the pipeline), the count starts from the position of the first decoding step.
    """

    target_compression_ratio: float = 0.5
    target_size: int = field(default=1, init=False)

    def __post_init__(self):
        super().__post_init__()
        assert 0 <= self.target_compression_ratio < 1, "target_compression_ratio must be between 0 and 1"
        self.tokens_seen: dict[int, int] = {}  # Per-layer number of tokens seen

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        layer_idx = int(module.layer_idx)
        q_len = kwargs["hidden_states"].shape[1]
        if is_prefilling(kwargs["cache_position"], q_len):
            self.tokens_seen[layer_idx] = q_len
        else:
            if layer_idx not in self.tokens_seen:
                # The prefill was not seen, count the tokens before the first position of this step
                positions = kwargs.get("position_ids")
                if positions is None:
                    positions = kwargs["cache_position"]
                self.tokens_seen[layer_idx] = int(positions[..., 0].max())
            self.tokens_seen[layer_idx] += q_len
        return super().forward_hook(module, input, kwargs, output)

    def _resolve_target_size(self, layer_idx: int) -> int:
        return max(1, int(self.tokens_seen[layer_idx] * (1 - self.target_compression_ratio)))

    def reset(self):
        super().reset()
        self.tokens_seen = {}
