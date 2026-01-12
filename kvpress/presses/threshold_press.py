# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values


@dataclass
class ThresholdPress(BasePress):
    """
    Wraps a ScorerPress and evicts keys/values with scores below a given threshold.

    Unlike most presses that use a fixed compression_ratio, ThresholdPress uses a score threshold
    to determine which KV pairs to evict. This allows for adaptive compression where the actual
    compression ratio depends on the input content.

    Importantly, this press can be used both during prefilling and during decoding (if decoding=True).

    A sliding window protects the most recent tokens from eviction, ensuring that recently
    generated tokens are always available for attention.

    Parameters
    ----------
    press : ScorerPress
        The underlying scorer press used to compute importance scores for each token.
    threshold : float, optional
        Tokens with scores below this threshold are evicted. The optimal threshold
        depends on the scorer press being used.
    sliding_window_size : int, default=128
        Number of recent tokens protected from eviction.
    decoding : bool, default=False
        If True, compression is also applied during the decoding phase (token generation).
        If False, compression only occurs during prefill.
    """

    press: ScorerPress
    threshold: Optional[float] = None
    sliding_window_size: int = 128
    decoding: bool = False
    scores_buffer: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    compression_ratios: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        """Average compression ratio across all layers (computed after forward pass)."""
        assert len(self.compression_ratios) > 0, "Forward pass must be run to compute the compression ratio"
        return sum(self.compression_ratios.values()) / len(self.compression_ratios)

    @compression_ratio.setter
    def compression_ratio(self, value):
        """Compression ratio is read-only since it depends on threshold and input content."""
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        cache_len = kwargs["cache_position"][-1] + 1
        prefilling = cache_len == q_len

        # Extract layer index as int for type safety
        layer_idx: int = module.layer_idx  # type: ignore[assignment]

        # Reset the scores buffer and compression ratios if we are in prefilling
        if prefilling and (layer_idx == 0):
            self.scores_buffer.clear()
            self.compression_ratios.clear()

        # Skip compression during decoding if not enabled
        if not prefilling and not self.decoding:
            return output

        # Compute importance scores for the new tokens using the underlying scorer press
        keys, values = extract_keys_and_values(cache, layer_idx)
        scores = self.press.score(module, hidden_states, keys[:, :, -q_len:], values[:, :, -q_len:], None, kwargs)

        # Accumulate scores in the buffer: reset during prefill, append during decoding
        if prefilling:
            self.scores_buffer[layer_idx] = scores
        else:
            self.scores_buffer[layer_idx] = torch.cat([self.scores_buffer[layer_idx], scores], dim=-1)

        # Once the buffer exceeds the sliding window, evict tokens with low scores
        if self.scores_buffer[layer_idx].shape[-1] > self.sliding_window_size:
            # Determine how many tokens have left the sliding window and can be evicted
            n_to_evict = self.scores_buffer[layer_idx].shape[-1] - self.sliding_window_size
            scores_to_evict = self.scores_buffer[layer_idx][..., :n_to_evict]
            self.scores_buffer[layer_idx] = self.scores_buffer[layer_idx][..., n_to_evict:]

            # Find tokens below threshold: returns (batch_idx, head_idx, token_idx) tuples
            new_masked_key_indices = list(torch.where(scores_to_evict < self.threshold))

            if len(new_masked_key_indices[0]) > 0:
                # Convert buffer-relative indices to cache-absolute indices
                # During prefill shift=0; during decoding we offset by the number of previously processed tokens
                shift = cache_len - scores_to_evict.shape[2] - self.sliding_window_size
                new_masked_key_indices[-1] += shift

                # Merge new masked indices with existing ones
                if module.masked_key_indices is None:
                    module.masked_key_indices = new_masked_key_indices  # type: ignore[assignment]
                else:
                    module.masked_key_indices = list(  # type: ignore[assignment]
                        torch.cat([i, new_i]) for i, new_i in zip(module.masked_key_indices, new_masked_key_indices)
                    )

        # Track compression ratio as the fraction of masked tokens
        if module.masked_key_indices is not None:
            bsz, num_key_value_heads, cache_len, _ = keys.shape
            n_masked = len(module.masked_key_indices[0])  # type: ignore[index]
            self.compression_ratios[layer_idx] = n_masked / (bsz * num_key_value_heads * cache_len)
        else:
            self.compression_ratios[layer_idx] = 0

        return output
