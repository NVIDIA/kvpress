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
       the decoding compression approach.
    
    2. **DuoAttentionPress**: Uses attention masking and streaming patterns that are 
       incompatible with iterative decoding compression.
    
    3. **FinchPress**: These require `position_embeddings` from kwargs
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
    token_buffer_size : int, default=1024
        Target number of tokens to keep after compression. Dynamically calculates 
        compression_ratio to achieve this buffer size.
    """

    base_press: ScorerPress
    compression_steps: int = 10
    token_buffer_size: int = 1024

    def __post_init__(self):
        # Buffer to store hidden states during decoding
        assert isinstance(self.base_press, ScorerPress), "DecodingPress requires a ScorerPress as input"
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
        """
        Delegate compression to the base press during decoding phase.
        
        Args:
            module: The transformer module being compressed
            hidden_states: Buffered hidden states from recent decoding steps (shape: [batch, buffer_len, hidden_dim])
            keys: Key cache from all previous steps including current (shape: [batch, n_heads, seq_len, head_dim])
            values: Value cache from all previous steps including current (shape: [batch, n_heads, seq_len, head_dim])
            attentions: Attention weights (shape varies by implementation)
            kwargs: Additional keyword arguments
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Compressed (keys, values) tensors
            
        Note:
            **Sequence length alignment**: During decoding compression, `hidden_states` contains the 
            buffered hidden states from recent decoding steps (buffer_len tokens), while `keys` and 
            `values` contain the full sequence history (seq_len tokens). The base press implementation
            should use keys.shape[2] for full sequence length calculations. The buffered hidden_states
            provide context for the most recent tokens when computing compression scores.
            
        Performance Note:
            It would be possible to speed up compression during decoding for certain scorer presses by
            storing existing scores in a buffer (e.g. KNormPress) and reusing them in subsequent compressions.
        """
        q_len = keys.shape[2]
        # compression_ratio = fraction of tokens to remove
        # n_kept = q_len * (1 - compression_ratio) = token_buffer_size
        # So: compression_ratio = 1 - (token_buffer_size / q_len)
        target_compression_ratio = max(0.0, 1.0 - (self.token_buffer_size / q_len))
        logger.debug(f"Compressing {q_len} with compression ratio {target_compression_ratio}")

        original_compression_ratio = self.base_press.compression_ratio
        self.base_press.compression_ratio = target_compression_ratio
        try:
            result = self.base_press.compress(
                module, hidden_states, keys, values, attentions, kwargs
            )
            return result
        finally:
            self.base_press.compression_ratio = original_compression_ratio

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

        # Get current cache size for this layer
        if isinstance(cache, QuantizedCache):
            current_cache_size = cache._quantized_key_cache[module.layer_idx].shape[2]
        else:
            current_cache_size = cache.key_cache[module.layer_idx].shape[2]

        # Add current hidden states to buffer
        self.hidden_states_buffer.append(hidden_states.detach().clone())
        self.step_count += 1

        # Apply compression if cache size exceeds token_buffer_size
        if current_cache_size > self.token_buffer_size:
            logger.debug(f"Applying decoding compression: cache_size={current_cache_size} > token_buffer_size={self.token_buffer_size}")

            # Get keys and values from cache
            if isinstance(cache, QuantizedCache):
                keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])
                values = cache._dequantize(cache._quantized_value_cache[module.layer_idx])
            else:
                keys = cache.key_cache[module.layer_idx]
                values = cache.value_cache[module.layer_idx]

            # Get attention weights from output
            attentions = output[1] if len(output) > 1 and output[1] is not None else None

            # Apply compression using buffered hidden states
            buffered_hidden_states = torch.cat(self.hidden_states_buffer, dim=1)
            keys, values = self.compress(module, buffered_hidden_states, keys, values, attentions, kwargs)
            logger.debug(f"Applied decoding compression: "
                         f"keys.shape: {keys.shape}, values.shape: {values.shape}")

            # Update cache with compressed keys and values
            if isinstance(cache, QuantizedCache):
                cache._quantized_key_cache[module.layer_idx] = cache._quantize(keys, axis=-1)
                cache._quantized_value_cache[module.layer_idx] = cache._quantize(values, axis=-1)
            else:
                cache.key_cache[module.layer_idx] = keys
                cache.value_cache[module.layer_idx] = values

            # Clear buffer after compression
            self.hidden_states_buffer.clear()
            self.step_count = 0

        return output

    def reset(self):
        """Reset the decoding press state."""
        self.hidden_states_buffer.clear()
        self.step_count = 0
