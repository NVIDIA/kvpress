# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel

from kvpress.presses.afterglow_press import afterglow
from kvpress.presses.base_press import BasePress, is_prefilling
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.utils import extract_keys_and_values, get_query_states

LM_HEAD_CHUNK_SIZE = 1024


@dataclass
class QueryAwareKVzipPlusPress(KVzipPress):
    """
    Single-pass, query-aware KVzip+ with Afterglow and a predictability gate.

    Instead of KVzip's context reconstruction, the KVzip+ scores are computed during prefill from the attention of the
    last `window_size` tokens, which contain the question (query-aware compression). Afterglow then lets each token
    inherit the decayed scores of the tokens before it (see AfterglowPress), scaled by u(t)^alpha with
    u(t) = 1 - p(x_t | x_<t) from the prefill's own next-token distribution, so that predictable continuations inherit
    less. As in KVzip, the budget is shared across layers and heads.

    Based on Afterglow (https://www.alphaxiv.org/abs/2610.afterglow-query-aware-kv-cache-eviction).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    layerwise : bool, default=False
        Whether to enable uniform compression ratios across layers.
    n_sink : int, default=4
        Number of initial tokens to preserve as attention sinks. They are not Afterglow sources.
    window_size : int, default=64
        Number of last tokens of the context used as queries (should cover the question).
    gamma : float, default=0.95
        Decay of the propagated scores per token.
    alpha : float, default=0.5
        Exponent of the predictability gate u(t)^alpha.
    """

    kvzip_plus_normalization: bool = True
    window_size: int = 64
    gamma: float = 0.95
    alpha: float = 0.5

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        assert 0 < self.gamma < 1, "gamma must be in (0, 1)"
        assert self.kvzip_plus_normalization, "QueryAwareKVzipPlusPress always uses the KVzip+ normalization"
        self._reset_internal_parameters()

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """Only score during prefill; the cache is compressed once all layers are scored (see __call__)."""
        hidden_states = kwargs["hidden_states"]
        if self.compression_ratio == 0 or not is_prefilling(kwargs["cache_position"], hidden_states.shape[1]):
            return output

        keys, values = extract_keys_and_values(kwargs["past_key_values"], module.layer_idx)
        bsz, num_key_value_heads, k_len, head_dim = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        window_size = min(self.window_size, k_len)

        cos, sin = kwargs["position_embeddings"]
        position_embeddings = (cos[:, -window_size:], sin[:, -window_size:])
        queries = get_query_states(module, hidden_states[:, -window_size:], position_embeddings)
        queries = queries.view(bsz, num_key_value_heads, num_key_value_groups, window_size, head_dim)

        attn_weights = torch.einsum("bhgqd,bhkd->bhgqk", queries.float(), keys.float()) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, window_size)
        attn_weights = attn_weights.softmax(dim=-1)

        # KVzip+ normalization: divide by ||h|| of the query, multiply by ||W_O v|| of the key
        attn_weights = attn_weights / hidden_states[:, -window_size:].float().norm(dim=-1)[:, None, None, :, None]
        attn_weights = attn_weights * self._compute_value_output_norm(module, values).float()[:, :, :, None, :]

        if self.score_val is None:
            n_layers = module.config.num_hidden_layers
            self.score_val = torch.zeros(n_layers, bsz, num_key_value_heads, k_len, device=keys.device)
        self.score_val[module.layer_idx] = attn_weights.amax(dim=(2, 3))  # max over group, queries
        return output

    @torch.no_grad()
    def unpredictability(self, model: PreTrainedModel, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        """u(t) = 1 - p(x_t | x_<t) from the final hidden states of the prefill, max-pooled over 3 tokens."""
        u = torch.ones(input_ids.shape, device=hidden_states.device)
        for start in range(0, input_ids.shape[1] - 1, LM_HEAD_CHUNK_SIZE):
            end = min(start + LM_HEAD_CHUNK_SIZE, input_ids.shape[1] - 1)
            logits = model.lm_head(hidden_states[:, start:end]).float()
            target = logits.gather(-1, input_ids[:, start + 1 : end + 1, None])[..., 0]
            u[:, start + 1 : end + 1] = 1 - (target - logits.logsumexp(dim=-1)).exp()
        return F.max_pool1d(u[:, None], 3, stride=1, padding=1)[:, 0]

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        captured = {}

        def capture_input_ids(module, args, kwargs):
            captured["input_ids"] = kwargs["input_ids"]

        def capture_hidden_states(module, input, output):
            captured["hidden_states"] = output

        hooks = [
            language_model.register_forward_pre_hook(capture_input_ids, with_kwargs=True),
            language_model.norm.register_forward_hook(capture_hidden_states),
        ]
        try:
            with BasePress.__call__(self, model):
                yield
            if self.score_val is not None:
                gate = self.unpredictability(model, captured["hidden_states"], captured["input_ids"])
                self.score_val = afterglow(self.score_val, self.gamma, self.n_sink, gate.pow(self.alpha)[:, None])
                self.compress_post(model)
        finally:
            for hook in hooks:
                hook.remove()
            self._reset_internal_parameters()
