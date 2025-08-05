# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Generator, List, cast

import torch
from torch import nn
from transformers import AutoTokenizer, Gemma3ForCausalLM, PreTrainedModel, PreTrainedTokenizer, QuantizedCache
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.llama.modeling_llama import rotate_half
from transformers.models.phi3.modeling_phi3 import Phi3Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from kvpress.presses.base_press import SUPPORTED_MODELS, BasePress

logger = logging.getLogger(__name__)


@dataclass
class KVzipPress(BasePress):
    """
    KVzip identifies the importance of KV pairs through context reconstruction,
    enabling effective query-agnostic KV cache compression.

    In this code, we implement KVzip with minimal changes to this repository.
    For a fully optimized implementation with actual compression,
    please refer to the original repository,
    which also provides a version without runtime compression overhead (at the cost of performance).
    Original repository (https://github.com/snu-mllab/KVzip).

    Based on KVzip (https://arxiv.org/abs/2505.23416).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    layerwise : bool, default=False
        Whether to enable uniform compression ratios across layers.
        When False, while the overall KV cache compression ratio is maintained,
        each layer has a different compression ratio.
    n_sink : int, default=4
        Number of initial tokens to preserve as attention sinks.
    """

    compression_ratio: float = 0.0
    layerwise: bool = False
    n_sink: int = 4

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        logger.warning(
            "KVzipPress requires multiple forward passes for chunked context reconstruction, "
            "resulting in a computational overhead of 2–3 times the initial prefilling cost. "
            "This significantly increases the overall prefilling time compared to other compression methods, "
            "which is inherent to the KVzip algorithm design."
        )
        self._reset_internal_parameters()

    def _reset_internal_parameters(self):
        self.context_length = 0
        self.prefix_length = 0

        self._suffix_ids = None
        self._context_ids = None
        self._cache = None

        self.score_val = None
        self.causal_mask_score = None
        self.start_idx = 0
        self.end_idx = 0

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager that handles both initial prefilling and KVzip scoring/compression.

        This overrides the base class __call__ method to implement the full KVzip algorithm:
        1. First yield: allows initial prefilling with context
        2. After yield: performs KVzip scoring and compression using context reconstruction
        """
        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning(f"Model {type(model)} not tested, supported models: {SUPPORTED_MODELS}")

        if isinstance(model, Gemma3ForCausalLM):
            logger.warning("Compression in Gemma3 is only applied to layer without sliding window attention")

        # Store model reference for later use
        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)

        # Get suffix_ids directly using tokenizer's chat template (do this once, not in hook)
        if tokenizer.chat_template is None:
            prefix_text = ""
            suffix_text = "\n"  # Default suffix for models without chat template
        else:
            # Use a dummy context to extract the question suffix from chat template
            dummy_context = "dummy context"
            separator = "\n" + "#" * len(dummy_context)
            temp_context = tokenizer.apply_chat_template(
                [{"role": "user", "content": dummy_context + separator}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            context, suffix_text = temp_context.split(separator)
            prefix_text = context.split(dummy_context)[0]

        # Tokenize suffix directly to ids
        self.prefix_length = tokenizer.encode(prefix_text, return_tensors="pt", add_special_tokens=False).shape[-1]
        self._suffix_ids = tokenizer.encode(suffix_text, return_tensors="pt", add_special_tokens=False)

        # Register hook to store the pointer for past_key_values
        original_forward = model.model.forward

        def wrapped_forward(model_self, *args, **kwargs):
            self._context_ids = kwargs["input_ids"]
            self._cache = kwargs["past_key_values"]
            return original_forward(*args, **kwargs)

        model.model.forward = MethodType(wrapped_forward, model.model)

        hooks = []
        try:
            yield
            model.model.forward = original_forward  # Restore original

            # After yield: KVzip scoring and compression phase
            if self.compression_ratio > 0 and self._context_ids is not None:
                # Now register attention hooks for compression
                for layer in model.model.layers:
                    if isinstance(model, Gemma3ForCausalLM) and layer.is_sliding:
                        continue
                    layer.self_attn.rotary_emb = model.model.rotary_emb
                    hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))

                self._perform_kvzip_compression(model, tokenizer)
        finally:
            for hook in hooks:
                hook.remove()
            self._reset_internal_parameters()

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Override the forward_hook of BasePress.
        During the forward_hook, KVzip only calculates importance scores,
        aggregates scores across all layers, and then performs compression.
        """

        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_value"]

        if isinstance(cache, QuantizedCache):
            keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])  # type: ignore[attr-defined]
            values = cache._dequantize(cache._quantized_value_cache[module.layer_idx])  # type: ignore[attr-defined]
        else:
            keys = cache.key_cache[module.layer_idx]
            values = cache.value_cache[module.layer_idx]

        # Compute importance scores for KV pairs in the prefilled context,
        # retaining only the originally prefilled KV pairs.
        keys, values = self.score_kvzip(module, hidden_states, keys, values, output[1], kwargs)

        if isinstance(cache, QuantizedCache):
            cache = cast(Any, cache)  # to ignore attr-defined style errors
            cache._quantized_key_cache[module.layer_idx] = cache._quantize(keys, axis=cache.axis_key)
            cache._quantized_value_cache[module.layer_idx] = cache._quantize(values, axis=cache.axis_value)
            cache.key_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache.value_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache._seen_tokens = keys.shape[2]
        else:
            cache.key_cache[module.layer_idx] = keys
            cache.value_cache[module.layer_idx] = values

        return output

    def _perform_kvzip_compression(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Perform the KVzip scoring and compression algorithm.
        """

        # Prepare chunked inputs for context reconstruction
        self.context_length = self._context_ids.shape[1]
        chunked_context_pairs = self.prepare(model, tokenizer)

        # Perform scoring through context reconstruction
        # Use the stored cache from the initial forward pass
        self.start_idx = self.prefix_length
        for prefill_ids, repeat_ids in chunked_context_pairs:
            self.end_idx = self.start_idx + prefill_ids.shape[1]
            # Pass the cache that was used in the initial forward pass
            model(
                input_ids=repeat_ids.to(model.device),
                past_key_values=self._cache,
                num_logits_to_keep=1,
            )
            self.start_idx = self.end_idx

        # Perform final compression
        self.compress_post(model)

    def _chunk_fn(self, ctx_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        """
        Chunk input tokens
        """
        ctx_len = ctx_ids.shape[1]
        if ctx_len > chunk_size:
            chunk_num = (ctx_len - 1) // chunk_size + 1

            chunked_input_ids = []
            for i in range(chunk_num):
                start = i * chunk_size
                end = (i + 1) * chunk_size
                a_ids = ctx_ids[:, start:end]
                if a_ids.shape[1] == 0:
                    continue
                chunked_input_ids.append(a_ids)
        else:
            chunked_input_ids = [ctx_ids]

        return chunked_input_ids

    def prepare(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        chunk_size: int = 2048,
        prev_postfix_size=8,
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare chunked inputs for KV importance scoring with context reconstruction
        """
        ctx_ids = self._context_ids[:, self.prefix_length :].to("cpu")

        # initialize score values
        self.score_val = torch.zeros(
            (
                model.config.num_hidden_layers,
                1,
                model.config.num_key_value_heads,
                self.context_length,
            ),  # only support batch size of 1
            dtype=model.dtype,
            device=model.device,
        )
        self.score_val[..., : self.n_sink] = 1.0

        chunked_context_pairs = []
        chunked_input_ids = self._chunk_fn(ctx_ids, chunk_size)
        for i, a_ids in enumerate(chunked_input_ids):
            if i == 0:
                prompt = "\n\nRepeat the previous context exactly."
                q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
            else:
                prompt = "\n\nRepeat the part of the previous context exactly, starting with"
                q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
                postfix_prev = chunked_input_ids[i - 1][:, -prev_postfix_size:]
                q_ids = torch.cat([q_ids, postfix_prev], dim=1)

            chunked_context_pairs.append((a_ids, torch.cat([q_ids, self._suffix_ids, a_ids], dim=1)))

        return chunked_context_pairs

    def _make_mask(self, attn_weights: torch.Tensor, window_size: int):
        """
        Define causal mask shared across layers
        """
        mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        self.causal_mask_score = mask[None, None, None, :, :]

    def _mask_causal(self, attn_weights: torch.Tensor, window_size: int):
        """
        Apply causal masking
        """
        if self.causal_mask_score is None:
            self._make_mask(attn_weights, window_size)
        elif self.causal_mask_score.size(-1) != window_size:
            self._make_mask(attn_weights, window_size)

        attn_weights[..., -window_size:, -window_size:] += self.causal_mask_score

    def score_kvzip(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the maximum cross-attention scores during context reconstruction,
        and return slices of the keys and values containing only the originally prefilled KV pairs,
        i.e., excluding KV pairs from repeated contexts.
        The computed scores are stored in self.score_val.
        """

        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        num_heads_kv = module.config.num_key_value_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // num_heads_kv

        if isinstance(module, Phi3Attention):
            qkv = module.qkv_proj(hidden_states)
            queries = qkv[..., : num_heads * head_dim]
        elif hasattr(module, "q_proj"):
            # Assume Llama-like attention layer
            queries = module.q_proj(hidden_states)
        else:
            raise NotImplementedError(f"KVzip not yet implemented for {module.__class__}.")

        queries = queries.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

        # Support for Qwen3 and Gemma3 QK norm
        if isinstance(module, (Qwen3Attention, Gemma3Attention)):
            queries = module.q_norm(queries)

        # Apply RoPE
        cos, sin = kwargs["position_embeddings"]
        queries = (queries * cos.unsqueeze(1)) + (rotate_half(queries) * sin.unsqueeze(1))
        queries = queries.view(bsz, num_heads_kv, num_key_value_groups, q_len, head_dim)

        # Subsample keys
        sink = min(self.n_sink, self.start_idx)
        ctx_len = self.end_idx - self.start_idx
        keys_subsampled = torch.cat(
            [
                keys[:, :, :sink],  # attention sink tokens (generally system prompt)
                keys[:, :, self.start_idx : self.end_idx],  # KV chunk in the cache
                keys[:, :, -q_len:],  # KV repeat chunk
            ],
            dim=2,
        )
        keys_subsampled = keys_subsampled.unsqueeze(2).transpose(-2, -1).contiguous()

        # Compute attention
        attn_weights = torch.matmul(queries, keys_subsampled) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights[..., sink : sink + ctx_len]
        scores = attn_weights.amax(dim=(-3, -2))  # max over group, q

        layer_idx = int(module.layer_idx)
        self.score_val[layer_idx][..., self.start_idx : self.end_idx] = scores  # update score

        # Retain the originally prefilled context KV pairs and exclude KV pairs from the repeated context
        keys, values = keys[:, :, : self.context_length], values[:, :, : self.context_length]
        return keys, values

    def compress_post(self, model: PreTrainedModel):
        """
        Obtain the indices of KV pairs to be evicted.
        Adopted from adakv_press.compress (fake compression). KVzip does not rely on safeguards.
        """
        if self.compression_ratio > 0:
            n_layer, bsz, num_key_value_heads, ctx_len = self.score_val.shape

            # calculate the pruned KV pairs across layers
            if self.layerwise:
                nl = int(num_key_value_heads * ctx_len * self.compression_ratio)
                n_pruned_layers = nl * torch.ones(n_layer, device=self.score_val.device, dtype=torch.int)
            else:
                score_sort = torch.sort(self.score_val.reshape(-1)).values  # ascending order
                n = max(int(len(score_sort) * self.compression_ratio) - 1, 0)
                thres = score_sort[n].item()

                n_pruned_layers = (self.score_val.reshape(n_layer, -1) <= thres).sum(-1)  # n_prune

            for layer in model.model.layers:
                if isinstance(model, Gemma3ForCausalLM) and layer.is_sliding:
                    # Skip layers with sliding window attention, only for Gemma3
                    continue
                module = layer.self_attn
                layer_idx = int(module.layer_idx)

                assert module.config._attn_implementation != "eager", "eager mode not supported"

                scores = self.score_val[layer_idx]

                # Compute bottom-k across heads
                n_pruned = n_pruned_layers[layer_idx]
                indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()

                # Save indices to mask during the attention mechanism. Please refer to attention_patch.py for details
                batch_indices = torch.arange(bsz, device=n_pruned.device).repeat_interleave(n_pruned)
                head_indices = indices // ctx_len
                seq_indices = indices % ctx_len
                module.masked_key_indices = (batch_indices, head_indices, seq_indices)
