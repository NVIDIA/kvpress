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
    
    Based on StreamingLLM (https://arxiv.org/abs/2309.17453), this method implements
    a sliding window approach that preserves the first few tokens (sink tokens) and
    the most recent tokens, while pruning tokens in the middle of the sequence.
    
    This approach is particularly effective for streaming applications where:
    - The beginning of the sequence contains important context (sink tokens)
    - Recent tokens are most relevant for current processing
    - Middle tokens can be safely discarded to maintain a fixed memory footprint
    
    The method works by:
    1. Always preserving the first n_sink tokens (attention sinks)
    2. Preserving the last n_local tokens (recent context)
    3. Pruning all tokens in between these two windows
    
    Note: The original StreamingLLM implementation also includes key rerotation.
    To achieve the full StreamingLLM behavior, combine with KeyRerotationPress:
    ```python
    press = KeyRerotationPress(press=StreamingLLMPress(compression_ratio, n_sink))
    ```
    """

    compression_ratio: float = 0.0
    """
    Fraction of key-value pairs to remove during compression.
    
    This determines how many recent tokens (n_local) to keep:
    n_local = (1 - compression_ratio) * seq_len - n_sink
    
    Higher compression ratios result in smaller local windows and more aggressive pruning.
    See ScorerPress.compression_ratio for detailed description.
    """
    
    n_sink: int = 4
    """
    Number of initial tokens to always preserve (sink tokens).
    
    These tokens at the beginning of the sequence are never pruned, regardless
    of the compression ratio. They serve as "attention sinks" that help maintain
    model stability and performance.
    
    Typical values:
    - 4: Default value, works well for most models
    - 0: No sink tokens (may hurt performance)
    - 8+: More conservative, preserves more initial context
    
    The sink tokens are preserved because language models often assign high
    attention weights to early tokens regardless of their semantic content.
    """

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
