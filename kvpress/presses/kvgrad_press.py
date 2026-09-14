# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Generator, Optional

import torch
from torch import nn
from transformers import Cache, PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicLayer
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.kvzip_press import KVzipPress
from kvpress.utils import extract_keys_and_values, get_prerope_query_states

logger = logging.getLogger(__name__)


class ReadOnlyDynamicLayer(DynamicLayer):
    """
    Cache layer whose `keys` and `values` are never updated after prefilling.
    `update` returns [prefilled | chunked replay] without storing it.
    `last_keys` keeps that concatenation, used to recompute the attention weights once the backward pass is over.
    """

    last_keys: Optional[torch.Tensor] = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs=None):
        self.last_keys = torch.cat([self.keys, key_states], dim=-2)
        return self.last_keys, torch.cat([self.values, value_states], dim=-2)


@contextmanager
def read_only_cache(cache: Cache) -> Generator[Cache, None, None]:
    """
    Temporarily replace the layers of `cache` by `ReadOnlyDynamicLayer` ones sharing the same
    keys and values, so that forward passes can read the prefilled cache without modifying it.
    """
    original_layers = list(cache.layers)
    read_only_layers: list = []
    for layer_idx in range(len(original_layers)):
        keys, values = extract_keys_and_values(cache, layer_idx)
        layer = ReadOnlyDynamicLayer()
        layer.is_initialized = True
        layer.dtype, layer.device = keys.dtype, keys.device
        layer.keys, layer.values = keys.detach(), values.detach()
        read_only_layers.append(layer)

    cache.layers = read_only_layers
    try:
        yield cache
    finally:
        cache.layers = original_layers


@contextmanager
def frozen_parameters(model: PreTrainedModel) -> Generator[None, None, None]:
    """
    Temporarily freeze the parameters of `model` to calculate gradients w.r.t. the activations only.
    The gradients w.r.t. the parameters are not needed for KVgrad scoring.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter in parameters:
            parameter.requires_grad_(True)


@dataclass
class KVgradPress(KVzipPress):
    """
    Rather than ranking KV pairs by a heuristic measured inside their own layer, KVgrad estimates
    the effect of each KV pair on the model’s final hidden states through input × gradient attribution.

    Following KVzip, we reconstruct the context and use the following score for each KV pair c at layer l:

        S_{l,c} = max_{p in P} A_{p,c}^l * ||v_c^l W_O^l||_2 * ||grad_{a_p^l} Phi||_2

    Here, A * ||v W_O|| measures the contribution of the KV pair to the pre-MLP residual stream a_p^l
    at replay probe token p, while ||grad_{a_p^l} Phi|| measures the sensitivity of the objective to that
    residual stream. The objective is Phi = sum_p ||h_p^L||^2, an aggregation of the last hidden states.
    Scores are kept per KV head, and the maximum also runs over GQA groups.

    Call order, the methods marked with a star being inherited from KVzipPress:

        prefill    __call__ -> __call__*
                     prefill the context outside inference mode, so that autograd can later
                     build a graph on the cached KV pairs
        replay     _perform_kvzip_compression -> prepare*, then _score_chunk per chunk
                     one forward pass over the probe prompt, during which the hooks set by
                     _register_gradient_hooks capture A (_replay_attention_weights), the last
                     hidden states h and the residual streams a
        gradient   Phi.backward()
                     populate the gradient of Phi w.r.t. every residual stream a_p^l
        scoring    _update_scores
                     A * ||v W_O|| * ||grad_a Phi|| -> score_val, per layer
        eviction   compress_post*
                     bottom-k of score_val -> masked_key_indices, read by attention_patch.py

    Based on KVgrad (https://openreview.net/forum?id=cg1wTCJjjk).

    Parameters
    ----------
    chunk_size : int, default=512
        Number of context tokens reconstructed by each replay pass. Smaller chunks give a finer
        gradient signal, at the cost of more forward and backward passes.
    compression_ratio, layerwise, n_sink
        See `KVzipPress`.
    """

    chunk_size: int = 512
    # KVgrad always weights the reconstruction attention by ||v W_O||, so the KVzip+ normalization
    # does not apply here.
    kvzip_plus_normalization: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        assert self.chunk_size > 0, "Chunk size must be positive"
        logger.warning(
            "KVgradPress replays the context with one forward and one backward pass per chunk, "
            "resulting in a computational overhead of 3-4 times the initial prefilling cost. "
            "This significantly increases the overall prefilling time compared to other compression methods, "
            "which is inherent to the KVgrad algorithm design."
        )
        self._reset_internal_parameters()

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context prefilling should not create graph, but tensors created under inference mode are
        permanently marked: they can neither be saved for backward nor updated in place afterwards.
        """
        with torch.inference_mode(False), torch.no_grad(), super().__call__(model):
            yield

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        # KVgrad scores the KV pairs with the hooks set by _register_gradient_hooks
        return output

    def _perform_kvzip_compression(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Score the KV pairs by replaying the context chunk by chunk, then compress.
        """
        self.context_length = self._context_ids.shape[1]
        self.start_idx = self.prefix_length

        # The whole scoring runs outside inference mode: prefilling may have been called from it,
        # and score_val would then be an inference tensor that cannot be updated afterwards.
        with torch.inference_mode(False), torch.enable_grad(), frozen_parameters(model):
            chunked_context_pairs = self.prepare(model, tokenizer, chunk_size=self.chunk_size)
            with read_only_cache(self._cache) as cache:
                for prefill_ids, repeat_ids in chunked_context_pairs:
                    self.end_idx = self.start_idx + prefill_ids.shape[1]
                    self._score_chunk(model, cache, repeat_ids)
                    self.start_idx = self.end_idx

        # Unlike KVzip attention weights, KVgrad scores are unbounded: sinks are given the highest one
        self.score_val[..., : self.n_sink] = self.score_val.amax() + 1.0

        self.compress_post(model)

    def _score_chunk(self, model: PreTrainedModel, cache: Cache, repeat_ids: torch.Tensor):
        """
        Replay a single chunk of the context and update the scores of its KV pairs.
        """
        captures: dict = {"hidden_states": None, "attn_weights": {}, "residuals": {}}
        hooks = self._register_gradient_hooks(model, cache, captures)
        try:
            # Embedding the tokens here guarantees a differentiable graph despite frozen parameters
            inputs_embeds = model.model.embed_tokens(repeat_ids.to(model.device))
            inputs_embeds = inputs_embeds.detach().requires_grad_(True)
            model.model(inputs_embeds=inputs_embeds, past_key_values=cache, use_cache=False)

            # Phi = sum_p ||h_p^L||^2, h being the output of the last decoder layer (pre final norm)
            Phi = (captures["hidden_states"].norm(dim=-1) ** 2).sum()
            Phi.backward()
        finally:
            for hook in hooks:
                hook.remove()

        with torch.no_grad():
            for layer_idx in captures["attn_weights"]:
                self._update_scores(model, cache, layer_idx, captures)

    def _register_gradient_hooks(self, model: PreTrainedModel, cache: Cache, captures: dict) -> list:
        """
        Register the hooks capturing the three quantities the scores are built from: the hidden
        states the objective is defined on, the reconstruction attention weights, and the pre-MLP
        residual stream of each layer, whose gradient measures how much the objective relies on it.
        """

        def objective_hook(module: nn.Module, args: tuple, output):
            captures["hidden_states"] = output[0] if isinstance(output, tuple) else output

        def attention_hook(module: nn.Module, args: tuple, kwargs: dict, output: list):
            with torch.no_grad():
                sink = min(self.n_sink, self.start_idx)
                ctx_len = self.end_idx - self.start_idx
                layer_idx = int(module.layer_idx)
                cache_layer = cache.layers[layer_idx]
                assert isinstance(cache_layer, ReadOnlyDynamicLayer)
                keys = cache_layer.last_keys
                attn_weights = self._replay_attention_weights(module, kwargs["hidden_states"], keys, kwargs)
                # Only the KV pairs of the chunk being reconstructed are scored
                captures["attn_weights"][layer_idx] = attn_weights[..., sink : sink + ctx_len].clone()

        def make_residual_hook(layer_idx: int) -> Callable:
            def residual_hook(module: nn.Module, args: tuple):
                residual = args[0]
                residual.retain_grad()
                captures["residuals"][layer_idx] = residual

            return residual_hook

        hooks = [model.model.layers[-1].register_forward_hook(objective_hook)]
        for layer_idx, layer in enumerate(model.model.layers):
            hooks.append(layer.self_attn.register_forward_hook(attention_hook, with_kwargs=True))
            # The input of the post-attention layer norm is the residual stream the attention
            # output has just been written to, i.e. the pre-MLP residual stream a
            hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(make_residual_hook(layer_idx)))
        return hooks

    def _replay_attention_weights(
        self, module: nn.Module, hidden_states: torch.Tensor, keys: torch.Tensor, kwargs: dict
    ) -> torch.Tensor:
        """
        Attention weights of the probe tokens over the attention sinks, the chunk being reconstructed
        and the probe tokens themselves, computed as in KVzipPress.score_kvzip.
        Shape (bsz, num_kv_heads, num_kv_groups, q_len, n_sink + chunk_len + q_len).
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads_kv = module.config.num_key_value_heads
        num_key_value_groups = module.config.num_attention_heads // num_heads_kv

        queries = get_prerope_query_states(module, hidden_states)
        cos, sin = kwargs["position_embeddings"]
        queries = (queries * cos.unsqueeze(1)) + (rotate_half(queries) * sin.unsqueeze(1))
        queries = queries.view(bsz, num_heads_kv, num_key_value_groups, q_len, module.head_dim)

        sink = min(self.n_sink, self.start_idx)
        keys_subsampled = torch.cat(
            [keys[:, :, :sink], keys[:, :, self.start_idx : self.end_idx], keys[:, :, -q_len:]], dim=2
        )
        keys_subsampled = keys_subsampled.unsqueeze(2).transpose(-2, -1).contiguous()

        attn_weights = torch.matmul(queries, keys_subsampled) / math.sqrt(module.head_dim)
        self._mask_causal(attn_weights, q_len)
        return nn.functional.softmax(attn_weights, dim=-1)

    def _update_scores(self, model: PreTrainedModel, cache: Cache, layer_idx: int, captures: dict):
        """
        Combine the reconstruction attention, the value signal magnitude and the gradient of the
        objective into the scores of the KV pairs of the current chunk.
        """
        module = model.model.layers[layer_idx].self_attn

        # A: reconstruction attention, with shape (bsz, num_kv_heads, num_kv_groups, q_len, chunk_len)
        attn_weights = captures["attn_weights"][layer_idx]

        # ||v W_O||: magnitude of the signal each KV pair writes into the residual stream,
        # with shape (bsz, num_kv_heads, num_kv_groups, chunk_len)
        num_heads_kv = module.config.num_key_value_heads
        num_key_value_groups = module.config.num_attention_heads // num_heads_kv
        Wo = module.o_proj.weight.transpose(0, 1)
        Wo = Wo.view(num_heads_kv, num_key_value_groups, module.head_dim, module.config.hidden_size)
        values = cache.layers[layer_idx].values[:, :, self.start_idx : self.end_idx]
        value_norm = torch.einsum("h g i j, b h t i -> b h g t j", Wo, values).norm(dim=-1)

        # ||grad_a Phi||: sensitivity of the objective to the residual stream of each probe token,
        # with shape (bsz, q_len)
        residual = captures["residuals"][layer_idx]
        assert residual.grad is not None, f"No gradient captured for layer {layer_idx}"
        gradient_norm = residual.grad.detach().float().norm(dim=-1)

        scores = attn_weights * value_norm[:, :, :, None, :]
        scores = scores * gradient_norm[:, None, None, :, None]
        self.score_val[layer_idx][..., self.start_idx : self.end_idx] = scores.amax(dim=(-3, -2))  # max over group, q
