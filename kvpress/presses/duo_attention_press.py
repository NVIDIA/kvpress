# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass, field
from io import StringIO

import numpy as np
import requests  # type: ignore[import-untyped]
import torch
from cachetools import LRUCache, cached  # type: ignore[import-untyped]
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from kvpress.presses.base_press import BasePress

PATTERNS_DICT = {
    "togethercomputer/Llama-2-7B-32K-Instruct": "Llama-2-7B-32K-Instruct/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
    "gradientai//Llama-3-8B-Instruct-Gradient-1048k": "Llama-3-8B-Instruct-Gradient-1048k/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
    "gradientai//Llama-3-8B-Instruct-Gradient-4194k": "Llama-3-8B-Instruct-Gradient-4194k/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10",  # noqa: E501
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct-v0.3/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
}

cache = LRUCache(maxsize=128)


@dataclass
class DuoAttentionPress(BasePress):
    """
    DuoAttention: Hybrid attention with retrieval and streaming heads.

    Splits attention heads into two types:
        - retrieval heads (use full KV cache) and
        - streaming heads (use only sink + recent tokens).
    Different heads have different attention patterns - some benefit from full context while others work well with
    limited context.

    Uses pre-computed attention patterns for supported models, falls back to
    on-the-fly computation for unsupported models.

    Based on DuoAttention (https://arxiv.org/abs/2410.10819).

    Parameters
    ----------
    head_compression_ratio : float, default=0.0
        Fraction of attention heads to convert to streaming heads.
        Controls balance between retrieval (full cache) and streaming (limited cache) heads.
    on_the_fly_scoring : bool, default=False
        Whether to compute attention patterns on-the-fly using random samples.
        If True, computes patterns instead of loading pre-computed ones.
    compression_ratio_ : float
        Actual compression ratio achieved (computed during forward pass).
    recent_size : int
        Size of recent token window for streaming heads (determined automatically).
    sink_size : int
        Number of initial tokens preserved for streaming heads (determined automatically).
    streaming_mask : torch.Tensor
        Binary mask indicating which heads are streaming heads.
    """

    head_compression_ratio: float = 0.0
    on_the_fly_scoring: bool = False
    compression_ratio_: float = field(init=False, default=None)
    recent_size: int = field(init=False, default=None)
    sink_size: int = field(init=False, default=None)
    streaming_mask: torch.Tensor = field(init=False, default=None)

    def __post_init_from_model__(self, model):
        """
        Initialize sink_size, recent_size, and streaming_mask from a model
        """
        # Load attention pattern from the DuoAttention repo
        if self.on_the_fly_scoring:
            self.sink_size, self.recent_size, head_scores = 128, 256, duo_attention_on_the_fly(model)
        else:
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
        if self.streaming_mask is None:
            raise ValueError(
                "Streaming mask not initialized. Make sure to call __post_init_from_model__ to initialize this press."
            )
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
    @cached(cache, key=lambda model: model.config.name_or_path)
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


@cached(cache, key=lambda model, num_samples=50, q_len=500: (model.config.name_or_path, num_samples, q_len))
def duo_attention_on_the_fly(model, num_samples=50, q_len=500):
    """
    New experimental method to quickly compute DuoAttention scores:
    - Compute the mean query and key on num_samples random samples from BookSum
    - Repeat the mean query and key q_len times and apply RoPE to get (Q, K)
    - Compute the attention weights for (Q[-1], K) and compute the "area under the cumulated attention curve"
    These scores could also be saved to avoid recomputing them but this method is still experimental
    """

    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads

    # Load data
    dataset = load_dataset("kmfoda/booksum", split="train").to_pandas()
    texts = dataset.sample(num_samples, random_state=42)["chapter"].tolist()

    # Initialize variables
    position_ids = torch.arange(q_len).unsqueeze(0)
    scores = torch.zeros((model.config.num_hidden_layers, num_key_value_heads))

    # Compute scores
    for text in texts:
        with torch.no_grad():
            # Compute hidden states
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            hidden_states = list(model(**inputs, output_hidden_states=True).hidden_states[:-1])

            for layer_idx, h in enumerate(hidden_states):
                module = model.model.layers[layer_idx]
                d = module.self_attn.head_dim
                h = module.input_layernorm(h)

                # Mean query
                q = module.self_attn.q_proj(h)
                q = q.view(1, q.shape[1], -1, d)
                if isinstance(module, (Gemma3Attention, Qwen3Attention)):
                    q = module.q_norm(q)
                q = q.mean(dim=1, keepdim=True)
                q = q.repeat(1, q_len, 1, 1).transpose(1, 2)

                # Mean key
                k = module.self_attn.k_proj(h)
                k = k.view(1, k.shape[1], -1, d)
                if isinstance(module, (Gemma3Attention, Qwen3Attention)):
                    k = module.k_norm(k)
                k = k.mean(dim=1, keepdim=True)
                k = k.repeat(1, q_len, 1, 1).transpose(1, 2)

                # Apply RoPE
                cos, sin = model.model.rotary_emb(h, position_ids.to(h.device))
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k = k.repeat_interleave(num_key_value_groups, dim=1)

                # Compute attention weights for the last token
                attn_weights = torch.matmul(q[:, :, -1:, :], k.transpose(2, 3)) / (d**0.5)
                attn_weights = attn_weights.softmax(dim=-1, dtype=torch.float32).squeeze()

                # Compute score: area under the cumulated attention curve
                s = torch.cumsum(attn_weights, dim=1).mean(1)
                s = s.view(-1, num_key_value_groups).mean(1)

                # Store the scores
                scores[layer_idx] += s.cpu() / num_samples
    return scores.numpy()
