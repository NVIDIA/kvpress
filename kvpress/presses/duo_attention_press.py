# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass, field
from io import StringIO

import numpy as np
import requests  # type: ignore[import-untyped]
import torch

from kvpress.presses.base_press import BasePress

PATTERNS_DICT = {
    "togethercomputer/Llama-2-7B-32K-Instruct": "Llama-2-7B-32K-Instruct/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
    "gradientai//Llama-3-8B-Instruct-Gradient-1048k": "Llama-3-8B-Instruct-Gradient-1048k/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
    "gradientai//Llama-3-8B-Instruct-Gradient-4194k": "Llama-3-8B-Instruct-Gradient-4194k/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10",  # noqa: E501
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct-v0.3/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
}


@dataclass
class DuoAttentionPress(BasePress):
    """
    Implements DuoAttention (https://arxiv.org/abs/2410.10819)

    Splits attention heads into two types:
    - Retrieval heads: use the full KV cache
    - Streaming heads: use only sink and recent tokens.

    Head classification is based on scores loaded from https://github.com/mit-han-lab/duo-attention/
    The higher the head_compression_ratio, the more streaming heads are used.
    """

    head_compression_ratio: float = 0.0
    compression_ratio_: float = field(init=False, default=None)
    recent_size: int = field(init=False, default=None)
    sink_size: int = field(init=False, default=None)
    streaming_mask: torch.Tensor = field(init=False, default=None)

    def __post_init_from_model__(self, model):
        """
        Initialize sink_size, recent_size, and streaming_mask from a model
        """
        # Load attention pattern from the DuoAttention repo
        self.sink_size, self.recent_size, head_scores = self.load_attention_pattern(model)

        # Define retrieval and streaming heads through a binary mask
        n_pruned = round(head_scores.size * self.head_compression_ratio)
        self.streaming_mask = torch.zeros(head_scores.shape, dtype=bool, device=model.device)
        if n_pruned > 0:
            indices = np.argsort(head_scores, axis=None)[:n_pruned]
            self.streaming_mask[np.unravel_index(indices, head_scores.shape)] = True

    @property
    def compression_ratio(self) -> float:
        assert self.compression_ratio_ is not None, "Forward pass must be run to compute the compression ratio"
        return self.compression_ratio_

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):

        assert module.config._attn_implementation != "eager", "eager mode not supported"
        q_len = hidden_states.shape[1]

        if (self.head_compression_ratio > 0) or (q_len > (self.sink_size + self.recent_size)):

            # Save indices to mask during the attention mechanism. Please refer to attention_patch.py for more details
            masked_keys = torch.zeros_like(keys[..., 0], dtype=torch.bool)
            masked_keys[:, self.streaming_mask[module.layer_idx], self.sink_size : -self.recent_size] = True
            module.masked_key_indices = torch.nonzero(masked_keys, as_tuple=True)

        # Compute the compression ratio
        self.compression_ratio_ = self.streaming_mask.float().mean().item()
        self.compression_ratio_ *= 1 - (self.sink_size + self.recent_size) / q_len

        return keys, values

    @staticmethod
    def load_attention_pattern(model):
        """
        Load the attention pattern from the DuoAttention repo
        """

        assert (
            model.config.name_or_path in PATTERNS_DICT
        ), f"Checkpoint {model.config.name_or_path} not in {list(PATTERNS_DICT.keys())}"
        base_url = "https://raw.githubusercontent.com/mit-han-lab/duo-attention/refs/heads/main/attn_patterns"
        url = f"{base_url}/{PATTERNS_DICT[model.config.name_or_path]}/"

        # Load config
        config = requests.get(url + "config.json").json()

        # Load head scores and clip as in duo_attn.utils.load_attn_pattern
        text = requests.get(url + "full_attention_heads.tsv").text
        head_scores = np.loadtxt(StringIO(text), dtype=float, delimiter="\t")
        head_scores = np.clip(head_scores, 0, 1)

        return config["sink_size"], config["recent_size"], head_scores

    @contextmanager
    def __call__(self, model):
        self.__post_init_from_model__(model)
        with super().__call__(model):
            yield
