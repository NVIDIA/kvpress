# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.cache_utils import QuantizedCache

from kvpress.presses.adakv_press import AdaKVPress
from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values

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
    compression_interval : int, default=512
        Number of decoding steps between compression, i.e. compression will be applied every compression_interval steps.
    target_size : int, default=2048
        Target number of tokens to keep after compression.
    hidden_states_buffer_size : int, default=256
        Maximum number of hidden states to keep before compression. Larger values use more GPU memory.
        Note: Some presses don't need buffered hidden states and can set this to 0 to use only the
        current hidden state for compression scoring.
    """

    base_press: ScorerPress | AdaKVPress
    compression_interval: int = 512
    target_size: int = 2048
    hidden_states_buffer_size: int = 256

    def __post_init__(self):
        # Buffer to store hidden states during decoding (per layer)
        assert isinstance(self.base_press, (ScorerPress, AdaKVPress)), "DecodingPress requires a ScorerPress as input"
        self.hidden_states_buffer = defaultdict(list)  # Per-layer buffer
        self.layer_step_counts = defaultdict(int)  # Track step count per layer

        assert self.compression_interval > 0, "compression_interval must be greater than 0"
        assert self.target_size > 0, "target_size must be greater than 0"

        if self.base_press.compression_ratio:
            logger.warning(
                f"compression_ratio is set for base press ({self.base_press.compression_ratio}). "
                f"This will be overridden by the decoding press."
            )

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
        k_len = keys.shape[2]
        target_compression_ratio = self._find_target_compression_ratio(k_len, self.target_size)
        logger.debug(f"Compressing {k_len} to {self.target_size} with ratio {target_compression_ratio}")

        original_compression_ratio = self.base_press.compression_ratio
        self.base_press.compression_ratio = target_compression_ratio
        result = self.base_press.compress(module, hidden_states, keys, values, attentions, kwargs)
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
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        layer_idx = module.layer_idx

        # Only operate during decoding phase (after prefilling)
        if kwargs["cache_position"][-1] <= q_len:
            # We're still in prefilling phase, don't do anything
            return output
        # print(f"Adding hidden states to buffer: {hidden_states.shape}")
        # Add current hidden states to buffer for this layer
        self.hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())

        # print(f"Layer step counts: {self.layer_step_counts[layer_idx]}")
        self.layer_step_counts[layer_idx] += 1

        # Apply compression if we've reached the compression step threshold
        if (self.layer_step_counts[layer_idx] >= self.compression_interval) or (q_len >= self.target_size):
            logger.debug(
                f"Applying decoding compression: layer_step_count ({self.layer_step_counts[layer_idx]}) >= compression_steps ({self.compression_interval})"  # noqa: E501
            )

            cache_layer = cache.layers[module.layer_idx]
            keys, values = extract_keys_and_values(cache, module.layer_idx)

            # Get attention weights from output
            attentions = output[1] if len(output) > 1 and output[1] is not None else None

            # Apply compression using buffered hidden states for this layer
            buffered_hidden_states = torch.cat(self.hidden_states_buffer[layer_idx], dim=1)
            keys, values = self.compress(module, buffered_hidden_states, keys, values, attentions, kwargs)
            logger.debug(f"Applied decoding compression: " f"keys.shape: {keys.shape}, values.shape: {values.shape}")

            # Update cache with compressed keys and values
            if isinstance(cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
                cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
                cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
                cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
                cache_layer.cumulative_length = keys.shape[2]
            else:
                cache_layer.keys = keys
                cache_layer.values = values

            # Reset step count and clear buffer for this layer
            self.layer_step_counts[layer_idx] = 0
            # Always clear the buffer after compression - otherwise there's a mismatch between
            # hidden states buffer and kv cache
            self.hidden_states_buffer[layer_idx] = []

        self.hidden_states_buffer[layer_idx] = (
            self.hidden_states_buffer[layer_idx][-self.hidden_states_buffer_size :]
            if self.hidden_states_buffer_size > 0
            else []
        )
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
            logger.warning(
                f"Binary search failed: q_len={q_len}, target={target_tokens}, got={final_n_kept}, ratio={ratio}"
            )

        return ratio
