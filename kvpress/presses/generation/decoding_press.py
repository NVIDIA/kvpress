# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
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
    

    Parameters
    ----------
    base_press : ScorerPress
        The scorer press used to compute importance scores for tokens.
    compression_steps : int, default=10
        Number of decoding steps between compression operations
    token_buffer_size : int, default=1024
        Target number of tokens to keep after compression.
    hidden_states_buffer_size : int, default=128
        Maximum number of hidden states to keep before compression. Larger values use more GPU memory.
        NoteSome presses don't need buffered hidden states and can set this to 0 to use only the
        current hidden state for compression scoring.
    """

    base_press: ScorerPress
    compression_steps: int = 128
    token_buffer_size: int = 1024
    hidden_states_buffer_size: int = 128

    def __post_init__(self):
        # Buffer to store hidden states during decoding (per layer)
        assert isinstance(self.base_press, ScorerPress), "DecodingPress requires a ScorerPress as input"
        self.hidden_states_buffer = defaultdict(list)  # Per-layer buffer
        self.layer_step_counts = defaultdict(int)  # Track step count per layer
        
        # Warn if compression happens before buffer is fully utilized
        # TODO: would it make sense to not reset the buffer?
        if self.hidden_states_buffer_size > 0 and self.compression_steps < self.hidden_states_buffer_size:
            logger.warning(f"compression_steps ({self.compression_steps}) < hidden_states_buffer_size ({self.hidden_states_buffer_size}). "
                          f"Buffer will be reset before reaching full capacity, potentially reducing compression quality.")

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
        target_compression_ratio = self._find_target_compression_ratio(q_len, self.token_buffer_size)
        logger.debug(f"Compressing {q_len} to {self.token_buffer_size} with ratio {target_compression_ratio}")

        original_compression_ratio = self.base_press.compression_ratio
        self.base_press.compression_ratio = target_compression_ratio
        result = self.base_press.compress(
            module, hidden_states, keys, values, attentions, kwargs
        )
        self.base_press.compression_ratio = original_compression_ratio
        return result

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
        layer_idx = module.layer_idx


        # Only operate during decoding phase (after prefilling)
        if kwargs["cache_position"][-1] <= q_len:
            # We're still in prefilling phase, don't do anything
            return output

        # Add current hidden states to buffer for this layer
        self.hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())

        self.layer_step_counts[layer_idx] += 1

        # Apply compression if we've reached the compression step threshold
        if self.layer_step_counts[layer_idx] >= self.compression_steps:
            logger.debug(f"Applying decoding compression: layer_step_count={self.layer_step_counts[layer_idx]} >= compression_steps={self.compression_steps}")

            # Get keys and values from cache
            if isinstance(cache, QuantizedCache):
                keys = cache._dequantize(cache._quantized_key_cache[layer_idx])
                values = cache._dequantize(cache._quantized_value_cache[layer_idx])
            else:
                keys = cache.key_cache[layer_idx]
                values = cache.value_cache[layer_idx]

            # Get attention weights from output
            attentions = output[1] if len(output) > 1 and output[1] is not None else None

            # Apply compression using buffered hidden states for this layer
            buffered_hidden_states = torch.cat(self.hidden_states_buffer[layer_idx], dim=1)
            keys, values = self.compress(module, buffered_hidden_states, keys, values, attentions, kwargs)
            logger.debug(f"Applied decoding compression: "
                         f"keys.shape: {keys.shape}, values.shape: {values.shape}")

            # Update cache with compressed keys and values
            if isinstance(cache, QuantizedCache):
                cache._quantized_key_cache[layer_idx] = cache._quantize(keys, axis=-1)
                cache._quantized_value_cache[layer_idx] = cache._quantize(values, axis=-1)
            else:
                cache.key_cache[layer_idx] = keys
                cache.value_cache[layer_idx] = values

            # Reset step count and clear buffer for this layer
            self.layer_step_counts[layer_idx] = 0
            self.hidden_states_buffer[layer_idx] = []

        self.hidden_states_buffer[layer_idx] = self.hidden_states_buffer[layer_idx][-self.hidden_states_buffer_size:]
        return output

    def reset(self):
        """Reset the decoding press state."""
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)

    def _find_target_compression_ratio(self, q_len: int, target_tokens: int) -> float:
        """
        Find the compression ratio that results in exactly target_tokens after int() rounding.

        Args:
            q_len: Current sequence length
            target_tokens: Desired number of tokens after compression

        Returns:
            Compression ratio that gives exactly target_tokens
        """
        if q_len <= target_tokens:
            return 0.0

        # Start with theoretical ratio
        ratio = 1.0 - (target_tokens / q_len)

        # Binary search to handle int() rounding
        low, high = 0.0, 1.0
        max_iterations = 20
        iteration = 0

        while iteration < max_iterations:
            n_kept = int(q_len * (1 - ratio))
            if n_kept == target_tokens:
                break
            elif n_kept > target_tokens:
                # Need more compression
                low = ratio
                ratio = (ratio + high) / 2
            else:
                # Need less compression
                high = ratio
                ratio = (low + ratio) / 2
            iteration += 1

        final_n_kept = int(q_len * (1 - ratio))
        if final_n_kept != target_tokens:
            logger.warning(f"Binary search failed: q_len={q_len}, target={target_tokens}, got={final_n_kept}, ratio={ratio}")

        return ratio
