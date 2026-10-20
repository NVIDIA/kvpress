# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Generator

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.kvzip_press import KVzipPress
from kvpress.utils import extract_keys_and_values, get_query_states


@dataclass
class EchoPress(KVzipPress):
    """
    EchoPress approximates the KVzip scores without replaying the context.

    KVzip scores a KV pair by the attention it receives when the model is asked to repeat its chunk,
    which costs one forward pass per chunk. As RoPE only depends on relative positions, the queries of
    the repeated tokens are approximated by the prefill queries rotated by the position offset of the
    repetition, and the keys of the repeated chunk are the cached keys up to the same rotation. The
    attention of this virtual repetition is thus computed from the prefill queries and the KV cache
    alone, in a hook on the prefill forward pass.

    A single reconstruction pass is then run on the first chunk, and its exact scores calibrate the
    virtual scores of the later chunks through a quantile mapping fitted independently for each layer
    and KV head. Eviction is inherited from KVzipPress.

    Only batch size 1 and a single prefill forward pass are supported. A Triton kernel for the virtual
    scoring step is available at https://github.com/ljwljwljwljw/kvpress/tree/echo-press-triton.

    Based on EchoPress (https://arxiv.org/abs/2610.00412).

    Parameters
    ----------
    compression_ratio, layerwise, n_sink, chunk_size
        See `KVzipPress`. `chunk_size` is both the size of the virtual chunks and the number of
        context tokens reconstructed by the single replay pass.
    """

    def __post_init__(self):
        super().__post_init__()
        assert not self.kvzip_plus_normalization, "EchoPress does not support the KVzip+ normalization"

    def _reset_internal_parameters(self):
        super()._reset_internal_parameters()
        self._pairs = []

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Score the context during prefilling, before KVzipPress replays the first chunk and compresses.
        """
        with super().__call__(model):
            hooks = []
            if self.compression_ratio > 0:
                prefill_hook = partial(self._prefill_hook, model=model)
                hooks = [
                    layer.self_attn.register_forward_hook(prefill_hook, with_kwargs=True)
                    for layer in model.model.layers
                ]
            try:
                yield
            finally:
                for hook in hooks:
                    hook.remove()

    def _prefill_hook(self, module: nn.Module, args: tuple, kwargs: dict, output: list, model: PreTrainedModel):
        """
        Compute the virtual scores of every chunk of the context for one layer.
        """
        layer_idx = int(module.layer_idx)
        if layer_idx == 0:
            self.context_length = self._context_ids.shape[1]
            self._pairs = self.prepare(model, self._tokenizer)

        hidden_states = kwargs["hidden_states"]
        cache = kwargs.get("past_key_values", None) or kwargs.get("past_key_value", None)
        keys, _ = extract_keys_and_values(cache, layer_idx)
        assert keys.shape[2] == self.context_length, "EchoPress requires a single prefill forward pass"
        queries = get_query_states(module, hidden_states, kwargs["position_embeddings"])

        start, inv_freq = self.prefix_length, model.model.rotary_emb.inv_freq
        for chunk_ids, repeat_ids in self._pairs:
            end = start + chunk_ids.shape[1]
            prompt_length = repeat_ids.shape[1] - chunk_ids.shape[1]
            self.score_val[layer_idx][..., start:end] = self._virtual_scores(
                queries, keys, start, end, prompt_length, inv_freq
            )
            start = end

    def _virtual_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        start: int,
        end: int,
        prompt_length: int,
        inv_freq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention weights of a virtual repetition of the chunk [start, end) over the attention sinks,
        the chunk and the repetition itself, maxed over the queries and the heads of each KV group as
        in KVzip. Shape (bsz, num_kv_heads, chunk_len).
        """
        bsz, num_heads, _, head_dim = queries.shape
        num_heads_kv = keys.shape[1]
        sink = min(self.n_sink, start)
        q_len = end - start

        # Queries of the repeated tokens: the prefill queries rotated by the position offset of the repetition
        q_copy = queries[:, :, start:end]
        angles = (self.context_length + prompt_length - start) * torch.cat([inv_freq, inv_freq])
        q_repeat = q_copy * angles.cos().to(q_copy.dtype) + rotate_half(q_copy) * angles.sin().to(q_copy.dtype)
        q_copy = q_copy.view(bsz, num_heads_kv, num_heads // num_heads_kv, q_len, head_dim)
        q_repeat = q_repeat.view(bsz, num_heads_kv, num_heads // num_heads_kv, q_len, head_dim)

        # Keys: attention sinks, KV chunk in the cache, and repeated chunk (its keys are the cached keys
        # up to the rotation, so the repeated queries attend to them as the prefill queries do)
        k_sink = keys[:, :, :sink].unsqueeze(2).transpose(-2, -1)
        k_chunk = keys[:, :, start:end].unsqueeze(2).transpose(-2, -1)
        attn_weights = torch.cat([q_repeat @ k_sink, q_repeat @ k_chunk, q_copy @ k_chunk], dim=-1)
        attn_weights = attn_weights / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        return attn_weights[..., sink : sink + q_len].amax(dim=(-3, -2))

    def _perform_kvzip_compression(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Replay the first chunk to score it exactly, calibrate the virtual scores of the later chunks
        on it, and compress.
        """
        chunk_ids, repeat_ids = self._pairs[0]
        self.start_idx, self.end_idx = self.prefix_length, self.prefix_length + chunk_ids.shape[1]
        virtual_scores = self.score_val[..., self.start_idx : self.end_idx].clone()

        hooks = [
            layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True) for layer in model.model.layers
        ]
        try:
            model.model(input_ids=repeat_ids.to(model.device), past_key_values=self._cache)
        finally:
            for hook in hooks:
                hook.remove()

        # Virtual and exact scores live on different scales: map the virtual scores of the later chunks
        # onto the exact scores of the first chunk, independently for each layer and KV head
        if len(self._pairs) > 1:
            exact_scores = self.score_val[..., self.start_idx : self.end_idx]
            later_scores = self.score_val[..., self.end_idx :]
            self.score_val[..., self.end_idx :] = self._quantile_map(virtual_scores, exact_scores, later_scores)

        self.compress_post(model)

    @staticmethod
    def _quantile_map(source: torch.Tensor, target: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Map values from the distribution of source to the distribution of target along the last
        dimension, by linear interpolation between the sorted quantiles of the two distributions.
        """
        source, target = source.float().sort(-1).values, target.float().sort(-1).values
        values = values.float().clamp(source[..., :1], source[..., -1:])
        idx = torch.searchsorted(source, values).clamp(1, source.shape[-1] - 1)
        x0, x1 = source.gather(-1, idx - 1), source.gather(-1, idx)
        y0, y1 = target.gather(-1, idx - 1), target.gather(-1, idx)
        w = ((values - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0, 1)
        return y0 + w * (y1 - y0)
