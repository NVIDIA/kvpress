# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.cache_utils import QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


@dataclass
class DecodingPress(BasePress):
    """
    A press that only operates during decoding phase and maintains a running buffer of hidden states.
    
    This press accumulates hidden states during decoding and applies compression every N steps
    using a scorer press to determine which tokens to keep.
    
    **COMPATIBILITY ISSUES AND LIMITATIONS:**
    
    **Presses that will NOT work with DecodingPress:**
    
    1. **AdaKVPress and CriticalAdaKVPress**: These presses use attention masking via 
       `module.masked_key_indices` which requires eager attention mode and conflicts with 
       the decoding compression approach. They also assume `q_len = hidden_states.shape[1]` 
       which is incorrect during decoding (should be `q_len = keys.shape[2]`).
    
    2. **DuoAttentionPress**: Uses attention masking and streaming patterns that are 
       incompatible with iterative decoding compression.
    
    3. **KeyRerotationPress and FinchPress**: These require `position_embeddings` from kwargs
       and perform RoPE operations that assume the full sequence context. During decoding,
       position embeddings may not be available or may be inconsistent with the compressed cache.
    
    4. **SnapKVPress, TOVAPress, ThinKPress**: These compute attention weights using 
       `compute_window_attention()` which requires position embeddings and assumes specific
       sequence structures that may be disrupted during iterative decoding compression.
    
    5. **SimLayerKVPress**: Uses lazy evaluation based on position embeddings and sequence
       patterns that may not work correctly with decoding compression.
    
    **Presses that SHOULD work with DecodingPress:**
    
    1. **ScorerPress with simple scorers**: Basic scoring functions like:
       - `expected_attention`: Uses hidden states only
       - `random`: No dependencies on sequence structure
    
    2. **KnormPress**: Only uses key norms, no position dependencies
    
    3. **StreamingLLMPress**: Simple sink-based scoring
    
    **Key Issues:**
    
    - **q_len mismatch**: Many presses use `q_len = hidden_states.shape[1]` but during 
      decoding, `hidden_states.shape[1] = 1` while the actual sequence length is 
      `keys.shape[2]`. To handle this, use `q_len = keys.shape[2]` during decoding.
    
    - **Position embeddings**: Presses requiring `kwargs["position_embeddings"]` may fail
      during decoding as position embeddings might not be available or consistent.
    
    - **Attention masking**: Presses that set `module.masked_key_indices` for attention
      masking are incompatible with the gather-based compression used in decoding.
    
    - **Sequence assumptions**: Presses that assume specific sequence structures or
      windowing patterns may break when applied iteratively during decoding.
    
    Parameters
    ----------
    base_press : ScorerPress
        The scorer press used to compute importance scores for tokens. Should use a simple
        scorer that doesn't depend on position embeddings or attention patterns.
    compression_steps : int, default=10
        Number of decoding steps between compression operations
    compression_ratio : float, default=0.5
        Fraction of tokens to remove during compression (0.0-1.0)
    """

    base_press: BasePress
    compression_steps: int = 10
    compression_ratio: float = 0.5

    def __post_init__(self):
        # Buffer to store hidden states during decoding
        self.hidden_states_buffer = []
        self.step_count = 0

    def compress(
            self,
            module: nn.Module,
            hidden_states: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attentions: torch.Tensor,
            kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.base_press.compress(
            module, hidden_states, keys, values, attentions, kwargs
        )

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Forward hook that manages decoding-specific compression logic.
        
        This hook:
        1. Detects when we're in decoding phase (not prefilling)
        2. Accumulates hidden states in a buffer
        3. Applies compression every N steps
        4. Clears the buffer after compression
        """
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_value"]
        q_len = hidden_states.shape[1]

        # Only operate during decoding phase (after prefilling)
        if kwargs["cache_position"][-1] <= q_len:
            # We're still in prefilling phase, don't do anything
            return output

        # We're in decoding phase

        # Add current hidden states to buffer
        self.hidden_states_buffer.append(hidden_states.detach().clone())
        self.step_count += 1

        # Apply compression every N steps
        if self.step_count >= self.compression_steps:
            logger.debug(f"Applying decoding compression at step {self.step_count}")

            # Get keys and values from cache
            if isinstance(cache, QuantizedCache):
                keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])
                values = cache._dequantize(cache._quantized_value_cache[module.layer_idx])
            else:
                keys = cache.key_cache[module.layer_idx]
                values = cache.value_cache[module.layer_idx]

            # Get attention weights from output
            attentions = output[1] if len(output) > 1 and output[1] is not None else None

            # Apply compression
            keys, values = self.compress(module, hidden_states, keys, values, attentions, kwargs)

            # Update cache with compressed keys and values
            if isinstance(cache, QuantizedCache):
                cache._quantized_key_cache[module.layer_idx] = cache._quantize(keys, axis=-1)
                cache._quantized_value_cache[module.layer_idx] = cache._quantize(values, axis=-1)
            else:
                cache.key_cache[module.layer_idx] = keys
                cache.value_cache[module.layer_idx] = values

            # Clear buffer and reset step count
            self.hidden_states_buffer.clear()
            self.step_count = 0

        return output

    def reset(self):
        """Reset the decoding press state."""
        self.hidden_states_buffer.clear()
        self.step_count = 0
