# Copyright (c) 2024 YuanFeng
#
# This file is part of the YuanFeng project and is licensed under the MIT License.
# SPDX-License-Identifier: MIT

from pickle import LIST
from attr import dataclass
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple
import torch


@dataclass
class MetaData:
    decoding_cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    max_seqlen_k: int = None
    cu_offset: torch.Tensor = None
    cu_head_offset: torch.Tensor = None
    head_lens: torch.Tensor = None 
    bsz: int = None 
    num_key_value_heads: int = None
    seen_tokens: int = None

    def _update_metadata_while_compressing(self, head_lens, cu_seqlens_k,max_seqlen_k):
        self.head_lens = head_lens
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_k = max_seqlen_k

    def _update_metadata_remove_n(self, n):
        self.max_seqlen_k -= n
        self.seen_tokens -= n
        self.head_lens -= n
        self.cu_seqlens_k -= self.cu_offset * n

    def _update_metadata(self, key_states):
        bs, head, seqlen, dim = key_states.shape

        self.max_seqlen_k += seqlen
        self.cu_seqlens_k += self.cu_offset * seqlen
        self.head_lens += seqlen
        self.seen_tokens += seqlen

    # init the metadata for the flattened cache during the prefilling phase
    def _init_metadata(self, key_states):

        """
        this method is used to initialize metadata for the flatten cache,
        input key_states is a regular key states with shape [bsz, num_key_value_heads, seqlen, head_dim]
        """

        bsz, num_key_value_heads, k_len, head_dim = key_states.shape
        k_lens = bsz *  num_key_value_heads * [k_len]
        _device = key_states.device
        max_seqlen_k = max(k_lens)

        head_seqlens_k = torch.tensor(k_lens, dtype=torch.int32, device=_device)
        cu_seqlens = torch.cumsum(head_seqlens_k, dim=0, dtype=torch.int32)
        cu_seqlens_k = torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=_device), cu_seqlens], dim=0)
        

        decoding_q_lens = bsz *  num_key_value_heads * [1] 
        decoding_head_seqlens_q = torch.tensor(decoding_q_lens, dtype=torch.int32,device=_device)
        decoding_cu_seqlens_q = torch.cumsum(decoding_head_seqlens_q, dim=0, dtype=torch.int32)
        decoding_cu_seqlens_q = torch.cat(
            [ torch.tensor([0], dtype=torch.int32, device=_device), decoding_cu_seqlens_q], dim=0)
        
        
        cu_offset = torch.arange(0, bsz * num_key_value_heads + 1, dtype=torch.int32, device=_device)
        cu_head_offset = torch.arange(1, bsz * num_key_value_heads + 1, dtype=torch.int32, device=_device)

        # init metadata
        self.decoding_cu_seqlens_q = decoding_cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_k = max_seqlen_k
        self.cu_offset = cu_offset
        self.cu_head_offset = cu_head_offset
        self.head_lens = head_seqlens_k
        self.bsz = bsz
        self.num_key_value_heads = num_key_value_heads
        self.seen_tokens = k_len

class DynamicCacheSplitHeadFlatten(Cache):
    
    """
    Flattened KV Cache Layout with a costomized update kernel
    """

    def __init__(self) ->None:
        super().__init__()
        self.key_cache: List[List[torch.Tensor]] = []
        self.value_cache: List[List[torch.Tensor]] = []
        self._seen_tokens = 0
        self.metadata_list:List[MetaData] = []

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
        # attn = cache_kwargs.get("attn", None)
        if len(self.key_cache) <= layer_idx:
            # prefilling
            # flatten key and value
            bs, head_num, seqlen, head_dim = key_states.shape
            flatten_key_cachee = key_states.reshape(bs* head_num* seqlen, head_dim)
            flatten_value_cache = value_states.reshape(bs* head_num* seqlen, head_dim)
            self.key_cache.append(flatten_key_cachee)
            self.value_cache.append(flatten_value_cache)
            meta_data = MetaData()
            meta_data._init_metadata(key_states)
            self.metadata_list.append(meta_data)
            # init metadata for flatten key states
            # self.metadata._init_metadata(key_states)
            self._seen_tokens = seqlen
        else:
            # decoding
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, head_dim = key_states.shape

            # TODO: Currently only support bs == 1
            assert bs == 1 , f"bs: {bs}"
            # NOTE: phase 2. we got [bs, head, seqlen, dim] as k, v input
            head_lens = self.metadata_list[layer_idx].head_lens
            cu_seqlens_k = self.metadata_list[layer_idx].cu_seqlens_k

            # TODO: wrap as a python interface
            from tiny_api_cuda import update_flatten_klenN_view
            new_key_cache = update_flatten_klenN_view(self.key_cache[layer_idx].view(-1, head_dim), key_states, head_lens, cu_seqlens_k)
            new_value_cache = update_flatten_klenN_view(self.value_cache[layer_idx].view(-1, head_dim), value_states, head_lens, cu_seqlens_k)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache

            # update metadata
            self.metadata_list[layer_idx]._update_metadata(key_states)
            self._seen_tokens += seqlen
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0

        # TODO: return 1 to means has content for now
        return 1

    def remove_tokens(self, n: int):
        raise NotImplementedError
        # for layer_idx in range(len(self.key_cache)):
        #     head_num = len(self.metadata_list[layer_idx].head_lens)
        #     cache_idx = torch.arange(0, self.key_cache[layer_idx].shape[0] - n * head_num, dtype=torch.int32, device=self.key_cache[layer_idx].device)
        #     pass 

        #     self.metadata_list[layer_idx]._update_metadata_remove_n(n)
        # self._seen_tokens -= n

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