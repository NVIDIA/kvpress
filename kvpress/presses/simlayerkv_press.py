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
    
    Based on SimLayerKV (https://arxiv.org/abs/2410.13846), this method uses a
    layer-wise approach to compression by identifying "lazy" layers that can
    work effectively with reduced KV cache sizes. The method dynamically
    determines which layers need full context and which can use streaming-style compression.
    
    The approach works by:
    1. Analyzing attention patterns of the last tokens in each layer
    2. Computing attention weight sums over initial and recent tokens
    3. Identifying layers as "lazy" if their attention is concentrated on recent/initial tokens
    4. Applying StreamingLLM-style compression to lazy layers (initial + recent tokens only)
    5. Using full KV cache for non-lazy layers that need complete context
    
    Layer classification is based on the lazy_threshold parameter:
    - If sum(attention_weights[last_tokens -> initial+recent]) > lazy_threshold: layer is lazy
    - Lazy layers use only n_initial + n_recent tokens
    - Non-lazy layers retain the full KV cache
    
    This adaptive approach optimizes memory usage by applying compression only
    where it won't significantly hurt performance, while preserving full context
    for layers that need it.
    
    Recommended lazy_threshold values from the official repository:
    - Llama3: 0.9
    - Llama2: 0.65  
    - Mistral: 0.8
    - Qwen: 0.85
    - Default: 1.0 (no compression, conservative)
    """

    lazy_threshold: float = 1.0
    """
    Threshold for identifying lazy layers based on attention concentration.
    
    This parameter controls which layers are considered "lazy" and can work
    effectively with reduced KV cache. A layer is classified as lazy if the
    sum of its last tokens' attention weights over initial and recent tokens
    exceeds this threshold.
    
    Lower thresholds:
    - Identify more layers as lazy (more aggressive compression)
    - May reduce memory usage significantly
    - Risk degrading performance if threshold is too low
    
    Higher thresholds:
    - Identify fewer layers as lazy (more conservative compression)
    - Preserve more full-context layers
    - Safer but may not achieve significant memory savings
    
    Model-specific recommended values:
    - 0.65: Llama2 models
    - 0.8: Mistral models  
    - 0.85: Qwen models
    - 0.9: Llama3 models
    - 1.0: No compression (default, very conservative)
    """
    
    n_last: int = 1
    """
    Number of last tokens to analyze for lazy layer identification.
    
    This parameter specifies how many of the most recent tokens are used
    to compute attention patterns for layer classification. The attention
    weights of these tokens are analyzed to determine if a layer focuses
    primarily on initial and recent tokens.
    
    The default value of 1 matches the SKLV-decode configuration and
    typically provides good layer classification results.
    """
    
    n_recent: int = 1024
    """
    Number of recent tokens to preserve in lazy layers.
    
    For layers identified as lazy, only the most recent n_recent tokens
    (plus n_initial initial tokens) are retained in the KV cache. This
    implements a StreamingLLM-style compression for these layers.
    
    Larger values:
    - Preserve more recent context for lazy layers
    - May improve performance but use more memory
    - Should be balanced against compression goals
    
    Smaller values:
    - Achieve more aggressive compression in lazy layers
    - May hurt performance if recent context is important
    - Provide greater memory savings
    
    The default of 1024 provides a reasonable balance for most applications.
    """
    
    n_initial: int = 4
    """
    Number of initial tokens to preserve in lazy layers (sink tokens).
    
    For layers identified as lazy, the first n_initial tokens are always
    preserved along with the n_recent recent tokens. These serve as
    attention sinks that help maintain model stability.
    
    See StreamingLLMPress.n_sink for detailed description of sink tokens.
    """

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
