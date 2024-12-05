# Copyright (c) 2024 YuanFeng
#
# This file is part of the YuanFeng project and is licensed under the MIT License.
# SPDX-License-Identifier: MIT

from transformers.cache_utils import Cache
from typing import List, Optional, Tuple
import torch


class DynamicCacheSplitHeadFlatten(Cache):
    
    """
    Flattened KV Cache Layout with a costomized update kernel
    """

    def __init__(self) ->None:
        super().__init__()
        self.key_cache: List[List[torch.Tensor]] = []
        self.value_cache: List[List[torch.Tensor]] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # NOTE: k, v = [head_num]( bs, 1, seqlen, dim)
        # each layer is a flatten layout like:
        # [bsz * (head_0_len + head_1_len + ...+ head_n_len) , dim]
        attn = cache_kwargs.get("attn", None)
        if len(self.key_cache) <= layer_idx:
            # prefilling
            # flatten key and value
            bs, head_num, seqlen, head_dim = key_states.shape
            flatten_key_cachee = key_states.reshape(bs* head_num* seqlen, head_dim)
            flatten_value_cache = value_states.reshape(bs* head_num* seqlen, head_dim)
            self.key_cache.append(flatten_key_cachee)
            self.value_cache.append(flatten_value_cache)

            # init metadata for flatten key states
            attn.metadata._init_metadata(key_states)
            self._seen_tokens = attn.metadata.seen_tokens
        else:
            # decoding
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, head_dim = key_states.shape

            # TODO: Currently only support bs == 1
            assert bs == 1 , f"bs: {bs}"
            # NOTE: phase 2. we got [bs, head, seqlen, dim] as k, v input
            head_lens = attn.metadata.head_lens
            cu_seqlens_k = attn.metadata.cu_seqlens_k

            # TODO: wrap as a python interface
            from tiny_api_cuda import update_flatten_klenN_view
            new_key_cache = update_flatten_klenN_view(self.key_cache[layer_idx].view(-1, head_dim), key_states, head_lens, cu_seqlens_k)
            new_value_cache = update_flatten_klenN_view(self.value_cache[layer_idx].view(-1, head_dim), value_states, head_lens, cu_seqlens_k)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache

            # update metadata
            attn.metadata._update_metadata(key_states)
            self._seen_tokens = attn.metadata.seen_tokens

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0

        # TODO: return 1 to means has content for now
        return 1

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        # print(f"to_legacy_cache")
        # legacy_cache = ()
        # for layer_idx in range(len(self)):
        #     legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx],),)
        # return legacy_cache
        raise NotImplementedError

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        # cache = cls()
        # print(f"from_legacy_cache past_key_values")
        # if past_key_values is not None:
        #     for layer_idx in range(len(past_key_values)):
        #         key_states, value_states = past_key_values[layer_idx]
        #         cache.key_cache.append(key_states)
        #         cache.value_cache.append(value_states)

        #         # TODO seen tokens  should be updated
        #         cache._seen_tokens = None
        # return cache
        raise NotImplementedError