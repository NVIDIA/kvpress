
# Copyright (c) 2024 YuanFeng
#
# This file is part of the YuanFeng project and is licensed under the MIT License.
# SPDX-License-Identifier: MIT

from transformers.utils import is_flash_attn_greater_or_equal_2_10
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
)


import logging
from typing import Optional, Tuple

import torch
from transformers import Cache

from kvpress.ada_cache import DynamicCacheSplitHeadFlatten

logger = logging.getLogger(__name__)


from transformers.utils import (
    logging,
    is_flash_attn_2_available,
)
if is_flash_attn_2_available():
    from flash_attn import  flash_attn_varlen_func

# replace the vanilla flash attention in the model with the flash_attn_varlen_func for head-specific compression support
def replace_var_flash_attn(model:str):
    from kvpress.ada_attn import AdaLlamaFlashAttention, AdaMistralFlashAttention
    from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES
    from transformers.models.mistral.modeling_mistral import MISTRAL_ATTENTION_CLASSES
    print(f"Replacing vanilla flash attention in {model} with flash_attn_varlen_func for head-specific compression support.")
    
    if "llama" in model.lower():
        LLAMA_ATTENTION_CLASSES["flash_attention_2"] = AdaLlamaFlashAttention
    elif "mistral" in model.lower():
        MISTRAL_ATTENTION_CLASSES["flash_attention_2"] = AdaMistralFlashAttention
    else:
        raise ValueError(f"Unsupported model: {model}")
    

class AdaLlamaFlashAttention(LlamaAttention):

    """
    Llama flash attention module for AdaKV. This module inherits from `LlamaAttention` as the weights of the module stays untouched.
    Utilizing the flash_attn_varlen_func from the flash_attn library to perform the attention operation with flattened KV Cache layout.
    """


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if not isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
            raise ValueError(
                "current implementation of `AdaKV` only supports `DynamicCacheSplitHeadFlatten` as the cache type."
            )
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        # query_states = query_states.transpose(1, 2)
        # key_states = key_states.transpose(1, 2)
        # value_states = value_states.transpose(1, 2)

        # dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)



        query_states = query_states.view(bsz,-1, self.num_key_value_groups,q_len ,self.head_dim)

        query_states = query_states.transpose(2, 3)
        query_states = query_states.reshape(-1,self.num_key_value_groups,self.head_dim)


        key_states = key_states.view(-1,1,self.head_dim)
        value_states = value_states.view(-1,1,self.head_dim)

        # get metadata for the flatten cache in the current layer
        current_layer_metadata = past_key_value.metadata_list[self.layer_idx]
        if q_len == 1:
            # get metadata for flatten query states during decoding phase
            cu_seqlens_q = current_layer_metadata.decoding_cu_seqlens_q
            max_seqlen_q = 1
        else:
            # init metadata for flatten query states during prefilling phase
            prefill_q_lens = bsz * self.num_heads//self.num_key_value_groups * [q_len]
            head_seqlens_q = torch.tensor(prefill_q_lens, dtype=torch.int32, device=query_states.device)
            cu_seqlens_q = torch.cumsum(head_seqlens_q, dim=0, dtype=torch.int32)
            cu_seqlens_q = torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=query_states.device), cu_seqlens_q], dim=0)
            max_seqlen_q = q_len
        
        cu_seqlens_k = current_layer_metadata.cu_seqlens_k
        max_seqlen_k = current_layer_metadata.max_seqlen_k


        attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q,
                                                cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)
        # TODO: support batch size > 1
        assert bsz == 1

        attn_output = attn_output.reshape(bsz, self.num_key_value_heads, q_len, self.num_key_value_groups, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class AdaMistralFlashAttention(MistralAttention):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        
        # used to store the metadata for the flatten cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        if not isinstance(past_key_value, DynamicCacheSplitHeadFlatten):
            raise ValueError(
                "current implementation of `AdaKV` only supports `DynamicCacheSplitHeadFlatten` as the cache type."
            )
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += cache_position[0]

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "attn": self}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)
        # dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)



        query_states = query_states.view(bsz,-1, self.num_key_value_groups,q_len ,self.head_dim)

        query_states = query_states.transpose(2, 3)
        query_states = query_states.reshape(-1,self.num_key_value_groups,self.head_dim)


        key_states = key_states.view(-1,1,self.head_dim)
        value_states = value_states.view(-1,1,self.head_dim)

        current_layer_metadata = past_key_value.metadata_list[self.layer_idx]

        if q_len == 1:
            # init metadata for flatten query states during decoding phase
            cu_seqlens_q = current_layer_metadata.decoding_cu_seqlens_q
            max_seqlen_q = 1
        else:
            # init metadata for flatten query states during prefilling phase
            prefill_q_lens = bsz * self.num_heads//self.num_key_value_groups * [q_len]
            head_seqlens_q = torch.tensor(prefill_q_lens, dtype=torch.int32, device=query_states.device)
            cu_seqlens_q = torch.cumsum(head_seqlens_q, dim=0, dtype=torch.int32)
            cu_seqlens_q = torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=query_states.device), cu_seqlens_q], dim=0)
            max_seqlen_q = q_len
            
        cu_seqlens_k = current_layer_metadata.cu_seqlens_k
        max_seqlen_k = current_layer_metadata.max_seqlen_k


        attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q,
                                                cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)
        # TODO: support batch size > 1
        assert bsz == 1

        attn_output = attn_output.reshape(bsz, self.num_key_value_heads, q_len, self.num_key_value_groups, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

