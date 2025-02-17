# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from io import StringIO
from typing import Generator
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import requests
import numpy as np
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)

PATTERNS_DICT = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10",  # noqa: E501
}


class DuoAttentionCache(Cache):
    """
    DuoAttentionCache maintains two caches: one for retrieval heads and one for streaming heads.
    Both caches are DynamicCache to align with the other presses, in particular we don't use the SinkCache
    for the streaming heads
    """

    def __init__(self):
        super().__init__()
        self.retrieval_cache = DynamicCache()
        self.streaming_cache = DynamicCache()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):

        if self.get_seq_length(layer_idx) == 0:
            # Prefilling
            return self.retrieval_cache.update(key_states, value_states, layer_idx, cache_kwargs)
        else:
            # Decoding
            mask = self.retrieval_mask[layer_idx]
            retrieval_key_states, retrieval_value_states = self.retrieval_cache.update(
                key_states[:, mask], value_states[:, mask], layer_idx, cache_kwargs
            )
            streaming_key_states, streaming_value_states = self.streaming_cache.update(
                key_states[:, ~mask], value_states[:, ~mask], layer_idx, cache_kwargs
            )

            return (
                (retrieval_key_states, streaming_key_states),
                (retrieval_value_states, streaming_value_states),
            )

    def get_seq_length(self, layer_idx=0):
        return self.retrieval_cache.get_seq_length(layer_idx)

    def __len__(self):
        return len(self.retrieval_cache)


@dataclass
class DuoAttentionPress(BasePress):
    """
    Efficient implementation of the DuoAttention paper (https://arxiv.org/abs/2410.10819)
    This press requires a custom cache (DuoAttentionCache) handling 2 caches for retrieval and streaming heads
    It also requires a custom implementation of the attention implementation to deal with 2 caches
    """

    head_compression_ratio: float = 0.0

    def forward_hook(self, module, input, kwargs, output):

        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_value"]
        q_len = hidden_states.shape[1]

        # No compression
        if self.head_compression_ratio == 0.0:
            return output

        # Short inputs
        if q_len <= (self.sink_size + self.recent_size):
            return output

        # Update cache
        layer_idx = module.layer_idx
        assert isinstance(cache, DuoAttentionCache), "Cache must be an instance of DuoAttentionCache"
        mask = self.retrieval_mask[layer_idx]
        module.retrieval_mask = mask

        # Keys
        keys = cache.retrieval_cache.key_cache[layer_idx]
        cache.retrieval_cache.key_cache[layer_idx] = keys[:, mask]
        cache.streaming_cache.key_cache.append(
            torch.cat([keys[:, ~mask, : self.sink_size, :], keys[:, ~mask, -self.recent_size :]], dim=2)
        )

        # Values
        values = cache.retrieval_cache.value_cache[layer_idx]
        cache.retrieval_cache.value_cache[layer_idx] = values[:, mask]
        cache.streaming_cache.value_cache.append(
            torch.cat([values[:, ~mask, : self.sink_size, :], values[:, ~mask, -self.recent_size :]], dim=2)
        )

        # Set the retrieval mask in the cache at the end of the prefilling as well as the compression ratio
        if module.layer_idx == (len(self.retrieval_mask) - 1):
            cache.retrieval_mask = self.retrieval_mask
            cr = self.retrieval_mask.float().mean().item()
            self.compression_ratio = (self.sink_size + self.recent_size) / q_len * cr + (1 - cr)
        return output

    @staticmethod
    def load_attention_pattern(model):
        assert (
            model.config.name_or_path in PATTERNS_DICT
        ), f"Checkpoint {model.config.name_or_path} not in {list(PATTERNS_DICT.keys())}"
        url = f"https://raw.githubusercontent.com/mit-han-lab/duo-attention/refs/heads/main/attn_patterns/{PATTERNS_DICT[model.config.name_or_path]}/"  # noqa: E501

        # Load config
        config = requests.get(url + "config.json").json()

        # Load head scores and clip (as in duo_attn.utils.load_attn_pattern)
        text = requests.get(url + "full_attention_heads.tsv").text
        head_scores = np.loadtxt(StringIO(text), dtype=float, delimiter="\t")
        head_scores = np.clip(head_scores, 0, 1)

        return config["sink_size"], config["recent_size"], head_scores

    @staticmethod
    def duo_attention_implementation(attn_implementation):

        attn = ALL_ATTENTION_FUNCTIONS[attn_implementation]

        def duo_attn(module, query, key, value, *args, **kwargs):
            if isinstance(key, torch.Tensor):
                # Prefilling
                return attn(module, query, key, value, *args, **kwargs)
            else:
                # Decoding
                retrieval_key, streaming_key = key
                retrieval_value, streaming_value = value
                mask = module.retrieval_mask
                query_mask = mask.repeat_interleave(module.num_key_value_groups)
                retrieval_query, streaming_query = query[:, query_mask], query[:, ~query_mask]

                # Get output for each cache
                out_retrieval, _ = attn(module, retrieval_query, retrieval_key, retrieval_value, *args, **kwargs)
                out_streaming, _ = attn(module, streaming_query, streaming_key, streaming_value, *args, **kwargs)
                out = torch.cat([out_retrieval, out_streaming], dim=2)

                # Reorder attention output in the correct order
                order = torch.argsort(torch.concat([torch.where(query_mask)[0], torch.where(~query_mask)[0]]))
                out = out[:, :, order]
                return out, None

        return duo_attn

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:

        # Update attention implementation
        attn_implementation = model.config._attn_implementation
        assert attn_implementation != "eager", "Eager attention is not supported for DuoAttentionPress"

        if not attn_implementation.startswith("duo_"):
            duo_attn = self.duo_attention_implementation(attn_implementation)
            ALL_ATTENTION_FUNCTIONS["duo_" + attn_implementation] = duo_attn
            model.config._attn_implementation = "duo_" + attn_implementation
            logger.warning("Updated attention implementation to duo_" + attn_implementation)

        # Load attention pattern from the DuoAttention repo
        self.sink_size, self.recent_size, head_scores = self.load_attention_pattern(model)

        # Define retrieval and streaming heads through a binary mask
        n_pruned = round(head_scores.size * self.head_compression_ratio)
        self.retrieval_mask = np.ones_like(head_scores, dtype=bool)
        self.retrieval_mask[np.unravel_index(np.argsort(head_scores, axis=None)[:n_pruned], head_scores.shape)] = False
        self.retrieval_mask = torch.from_numpy(self.retrieval_mask).to(device=model.device)

        # Register hooks
        with super().__call__(model):
            yield
