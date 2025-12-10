# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
import torch.nn as nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values


@dataclass
class ThresholdPress(BasePress):
    """
    Compute the scores of a ScorerPress and evict key and values with scores below a given threshold.
    An sliding window is used to avoid evicting the most recent keys and values.
    If decoding is enabled (default False), the eviction is also applied during the decoding phase.
    """

    press: ScorerPress
    threshold: float = None
    sliding_window_size: int = 128
    decoding: bool = False

    def __post_init__(self):
        self.scores_buffer = {}
        self.compression_ratios = {}

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        assert len(self.compression_ratios) > 0, "Forward pass must be run to compute the compression ratio"
        return sum(self.compression_ratios.values()) / len(self.compression_ratios)

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        cache_len = kwargs["cache_position"][-1] + 1

        # Reinitialize the score buffer and compression ratios at the start of a new sequence
        prefilling = cache_len == q_len
        if (module.layer_idx == 0) & prefilling:
            self.__post_init__()

        # Optionally skip the compression during decoding phase
        if not prefilling and not self.decoding:
            return output

        # Get scores for the tokens associated with the hidden states
        keys, values = extract_keys_and_values(cache, module.layer_idx)
        scores = self.press.score(module, hidden_states, keys[:, :, -q_len:], values[:, :, -q_len:], None, kwargs)

        # Update scores buffer
        if prefilling:
            self.scores_buffer[module.layer_idx] = scores
        else:
            self.scores_buffer[module.layer_idx] = torch.cat([self.scores_buffer[module.layer_idx], scores], dim=-1)

        # Update masked key indices if the scores buffer is full
        if self.scores_buffer[module.layer_idx].shape[-1] > self.sliding_window_size:
            scores_to_evict = self.scores_buffer[module.layer_idx][..., : -self.sliding_window_size]
            self.scores_buffer[module.layer_idx] = self.scores_buffer[module.layer_idx][
                ..., -self.sliding_window_size :
            ]
            new_masked_key_indices = list(torch.where(scores_to_evict < self.threshold))

            # Only update the masked key indices if there are some KV pairs to evict
            if len(new_masked_key_indices[0]) > 0:
                shift = cache_len - scores_to_evict.shape[2] - self.sliding_window_size
                new_masked_key_indices[-1] += shift

                if module.masked_key_indices is None:
                    module.masked_key_indices = new_masked_key_indices  # type: ignore[assignment]
                else:
                    module.masked_key_indices = list(  # type: ignore[assignment]
                        torch.cat([i, new_i]) for i, new_i in zip(module.masked_key_indices, new_masked_key_indices)
                    )

        # Update compression ratio
        if module.masked_key_indices is not None:
            bsz, num_key_value_heads, cache_len, _ = keys.shape
            self.compression_ratios[module.layer_idx] = len(module.masked_key_indices[0]) / (  # type: ignore[index]
                bsz * num_key_value_heads * cache_len
            )
        else:
            self.compression_ratios[module.layer_idx] = 0

        return output
