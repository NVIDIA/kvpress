# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, QuantizedCache

from .base_press import BasePress
from .scorer_press import ScorerPress

logger = logging.getLogger(__name__)


class DecodingPress(BasePress):
    """
    A press that only operates during decoding phase and maintains a running buffer of hidden states.
    
    This press accumulates hidden states during decoding and applies compression every N steps
    using a scorer press to determine which tokens to keep.
    
    Parameters
    ----------
    scorer_press : ScorerPress
        The scorer press used to compute importance scores for tokens
    compression_steps : int, default=10
        Number of decoding steps between compression operations
    compression_ratio : float, default=0.5
        Fraction of tokens to remove during compression (0.0-1.0)
    """
    
    def __init__(
        self,
        scorer_press: ScorerPress,
        compression_steps: int = 10,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.scorer_press = scorer_press
        self.compression_steps = compression_steps
        self.compression_ratio = compression_ratio
        
        # Buffer to store hidden states during decoding
        self.hidden_states_buffer = []
        self.step_count = 0
        self.is_decoding = False
        
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
        Compress KV cache during decoding using accumulated hidden states.
        
        Parameters
        ----------
        module : nn.Module
            Transformer attention layer
        hidden_states : torch.Tensor
            Current hidden states
        keys : torch.Tensor
            Keys of the cache (unquantized)
        values : torch.Tensor
            Values of the cache (unquantized)
        attentions : torch.Tensor
            Attention weights of the layer
        kwargs : dict
            Keyword arguments from the forward pass
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated keys and values
        """
        # Use the scorer press to compute importance scores
        # The scorer press will use the accumulated hidden states for scoring
        return self.scorer_press.compress(
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
        self.is_decoding = True
        
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
            compressed_keys, compressed_values = self.compress(
                module, hidden_states, keys, values, attentions, kwargs
            )
            
            # Update cache with compressed keys and values
            if isinstance(cache, QuantizedCache):
                cache._quantized_key_cache[module.layer_idx] = cache._quantize(compressed_keys, axis=-1)
                cache._quantized_value_cache[module.layer_idx] = cache._quantize(compressed_values, axis=-1)
            else:
                cache.key_cache[module.layer_idx] = compressed_keys
                cache.value_cache[module.layer_idx] = compressed_values
                
            # Clear buffer and reset step count
            self.hidden_states_buffer.clear()
            self.step_count = 0
            
        return output
        
    def reset(self):
        """Reset the decoding press state."""
        self.hidden_states_buffer.clear()
        self.step_count = 0
        self.is_decoding = False
