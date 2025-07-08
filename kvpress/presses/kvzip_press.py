# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import math

import torch
from torch import nn
from transformers import PreTrainedTokenizer, PreTrainedModel, Gemma3ForCausalLM
from transformers import QuantizedCache
from transformers.models.llama.modeling_llama import rotate_half
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.phi3.modeling_phi3 import Phi3Attention

from kvpress.presses.base_press import BasePress


@dataclass
class KVzipPress(BasePress):
    """
    KVzip identifies the importance of KV pairs through context reconstruction, 
    enabling effective query-agnostic KV cache compression.

    In this code, we implement KVzip with minimal changes to this repository.    
    For a fully optimized implementation with actual compression, 
    please refer to the original repository, 
    which also provides a version without runtime compression overhead (at the cost of performance).
    Original repo (https://github.com/snu-mllab/KVzip). 

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
    
    # The following variables are automatically set in pipeline.py    
    do_compress: bool = False
    context: str = None
    suffix: str = None    
    context_length: int = 0
    start_idx: int = 0    
    end_idx: int = 0    
    score_val: torch.Tensor = None
    causal_mask_score: torch.Tensor = None
        

    def _chunk_fn(self, ctx_ids: torch.Tensor, chunk_size: int):
        """ Chunk input tokens
        """
        ctx_len = ctx_ids.shape[1]
        if ctx_len > chunk_size:
            chunk_num = (ctx_len - 1) // chunk_size + 1

            input_ids = []
            for i in range(chunk_num):
                start = i * chunk_size
                end = (i + 1) * chunk_size
                a_ids = ctx_ids[:, start:end]
                if a_ids.shape[1] == 0:
                    continue
                input_ids.append(a_ids)
        else:
            input_ids = [ctx_ids]

        return input_ids


    def prepare(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        full_context_length: int,
        chunk_size: int = 2048,
        prev_postfix_size=8,
    ):
        """ Prepare chunked inputs for KV importance scoring with context reconstruction
            return: List[torch.Tensor]
        """

        ctx_ids = tokenizer.encode(self.context, return_tensors="pt", add_special_tokens=False)
        suffix_ids = tokenizer.encode(self.suffix, return_tensors="pt", add_special_tokens=False)

        self.context_length = full_context_length
        self.start_idx = full_context_length - ctx_ids.shape[1]  # the length of a system prompt

        # initialize score values 
        n_layers, n_kv = model.config.num_hidden_layers, model.config.num_key_value_heads
        self.score_val = torch.zeros((n_layers, 1, n_kv, full_context_length),  # only support batch size of 1
                        dtype=model.dtype, 
                        device=model.device)
        self.score_val[..., :self.n_sink] = 1.
        
                
        input_ids = []
        chunked_inputs = self._chunk_fn(ctx_ids, chunk_size)
        for i, a_ids in enumerate(chunked_inputs):
            if i == 0:
                prompt = f"\n\nRepeat the previous context exactly."
                q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
            else:
                prompt = f"\n\nRepeat the part of the previous context exactly, starting with"
                q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
                postfix_prev = chunked_inputs[i - 1][:, -prev_postfix_size:]
                q_ids = torch.cat([q_ids, postfix_prev], dim=1)

            input_ids.append((a_ids, torch.cat([q_ids, suffix_ids, a_ids], dim=1)))

        return input_ids
    

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Overwrite the forward_hook of BasePress
        """

        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_value"]
        q_len = hidden_states.shape[1]

        # Modified condition 
        if not self.do_compress:
            return output

        if isinstance(cache, QuantizedCache):
            keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])
            values = cache._dequantize(cache._quantized_value_cache[module.layer_idx])
        else:
            keys = cache.key_cache[module.layer_idx]
            values = cache.value_cache[module.layer_idx]

        # Only do scoring. Compression is done afterward by press.compress(self.model) in pipeline.py.
        keys, values = self.score(module, hidden_states, keys, values, output[1], kwargs)

        if isinstance(cache, QuantizedCache):
            cache._quantized_key_cache[module.layer_idx] = cache._quantize(keys, axis=cache.axis_key)
            cache._quantized_value_cache[module.layer_idx] = cache._quantize(values, axis=cache.axis_value)
            cache.key_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache.value_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache._seen_tokens = keys.shape[2]
        else:
            cache.key_cache[module.layer_idx] = keys
            cache.value_cache[module.layer_idx] = values

        return output


    def _make_mask(self, attn_weights: torch.Tensor, window_size: int):
        """ 
        Define causal mask shared across layers
        """
        mask = torch.full((window_size, window_size),
                          torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        self.causal_mask_score = mask[None, None, None, :, :]
        

    def _mask_causal(self, attn_weights: torch.Tensor, window_size: int):
        """ 
        Apply causal maksing
        """
        if self.causal_mask_score is None:
            self._make_mask(attn_weights, window_size)
        elif self.causal_mask_score.size(-1) != window_size:
            self._make_mask(attn_weights, window_size)

        attn_weights[..., -window_size:, -window_size:] += self.causal_mask_score
        

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute the max cross-attention scores during the context reconstruction.
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
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")

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
                keys[:, :, self.start_idx:self.end_idx],  # KV chunk in the cache
                keys[:, :, -q_len:],  # KV repeat chunk
            ],
            dim=2)        
        keys_subsampled = keys_subsampled.unsqueeze(2).transpose(-2, -1).contiguous()

        # Compute attention
        attn_weights = torch.matmul(queries, keys_subsampled) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        attn_weights = attn_weights[..., sink:sink + ctx_len]
        scores = attn_weights.amax(dim=(-3, -2))  # max over group, q
        
        self.score_val[module.layer_idx][..., self.start_idx:self.end_idx] = scores

        keys, values = keys[:,:,:self.context_length], values[:,:,:self.context_length]        
        return keys, values
            

    def compress(self, model: PreTrainedModel):
        """
        Adopted from adakv_press.compress (fake compression). KVzip does not rely on safeguards. 
        """
        if self.compression_ratio > 0:
            n_layer, bsz, num_key_value_heads, ctx_len = self.score_val.shape

            # calculate the pruned KV pairs across layers
            if self.layerwise:
                n_pruned = int(num_key_value_heads * ctx_len * self.compression_ratio)
                n_pruned_layers = n_pruned * torch.ones(n_layer, device=self.score_val.device, dtype=torch.int)
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
                
                assert module.config._attn_implementation != "eager", "eager mode not supported"       
         
                scores = self.score_val[module.layer_idx]

                # Compute bottom-k across heads
                n_pruned = n_pruned_layers[module.layer_idx]
                indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()

                # Save indices to mask during the attention mechanism. Please refer to attention_patch.py for more details
                batch_indices = torch.arange(bsz, device=n_pruned.device).repeat_interleave(n_pruned)
                head_indices = indices // ctx_len
                seq_indices = indices % ctx_len
                module.masked_key_indices = (batch_indices, head_indices, seq_indices)
                
