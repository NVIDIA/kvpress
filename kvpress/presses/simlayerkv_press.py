# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.snapkv_press import SnapKVPress

logger = logging.getLogger(__name__)


@dataclass
class SimLayerKVPress(BasePress):
    """
    SimLayerKV: Similarity-based layer-wise KV cache compression.

    Identifies "lazy" layers that can work effectively with reduced KV cache sizes.
    If a layer is considered "lazy", we only keep the initial and recent KV pairs.
    Otherwise, we keep all KV pairs.

    Recommended lazy_threshold values: Llama3 (0.9), Llama2 (0.65), Mistral (0.8), Qwen (0.85).

    Based on SimLayerKV (https://arxiv.org/abs/2410.13846).

    Parameters
    ----------
    lazy_threshold : float, default=1.0
        Threshold for identifying lazy layers based on attention concentration.
        Layer is lazy if sum(attention_weights[last_tokens -> initial+recent]) > threshold.
        Lower values identify more layers as lazy (more aggressive compression).
    n_last : int, default=1
        Number of last tokens to analyze for lazy layer identification.
    n_recent : int, default=1024
        Number of recent tokens to preserve in lazy layers.
    n_initial : int, default=4
        Number of initial tokens to preserve in lazy layers (sink tokens).
    """

    lazy_threshold: float = 1.0
    n_last: int = 1  # n_last=1 to match SKLV-decode
    n_recent: int = 1024
    n_initial: int = 4

    def __post_init__(self):
        assert 0.0 <= self.lazy_threshold <= 1.0, "lazy_threshold should be in [0, 1]"
        self.compression_ratios = []

    def is_lazy(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> bool:
        """
        Compute the attention weights of the last tokens over the initial and recent tokens.
        The layer is considered lazy if the sum of these attention weights is above the lazy_threshold.
        """

        attn_weights = SnapKVPress.compute_window_attention(
            module, hidden_states, keys, self.n_last, position_embeddings
        )
        attn_weights = attn_weights.mean((0, 1, 2))  # mean over bsz, heads and window size
        score = attn_weights[: self.n_initial].sum() + attn_weights[-self.n_recent :].sum()
        return score.item() > self.lazy_threshold

    @property
    def compression_ratio(self):
        if len(self.compression_ratios) > 0:
            return sum(self.compression_ratios) / len(self.compression_ratios)
        else:
            raise ValueError("Forward pass must be run to compute the compression ratio")

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Initialize the compression ratios
        if module.layer_idx == 0:
            self.compression_ratios = []

        # Check if compression is needed
        q_len = hidden_states.shape[1]
        min_length = self.n_initial + self.n_recent + self.n_last

        if q_len <= min_length:
            logger.warning(f"Sequence length is shorter than {min_length}: no compression applied")

        if (self.lazy_threshold == 1.0) or (q_len <= min_length):
            self.compression_ratios.append(0.0)
            return keys, values

        # Compression
        if self.is_lazy(module, hidden_states, keys, kwargs["position_embeddings"]):
            # If layer is lazy, only keep the initial and recent KV pairs
            keys = torch.cat([keys[:, :, : self.n_initial], keys[:, :, -self.n_recent + self.n_last :]], dim=2)
            values = torch.cat([values[:, :, : self.n_initial], values[:, :, -self.n_recent + self.n_last :]], dim=2)
            self.compression_ratios.append((q_len - self.n_initial - self.n_recent + 1) / q_len)
        else:
            self.compression_ratios.append(0.0)

        return keys, values
