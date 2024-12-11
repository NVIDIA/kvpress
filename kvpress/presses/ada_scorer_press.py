# Author: Yuan Feng
# Paper: [Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](https://arxiv.org/abs/2407.11550)



from functools import cache
import logging
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)


@dataclass
class AdaScorerPress(BasePress):
    """
    The press method defines a scoring mechanism within a head-specific paradigm, where the cache is adaptively pruned across all heads. 
    For more details, refer to the (Ada-KV)[https://arxiv.org/abs/2407.11550] paper.

    Any subclass of AdaScorerPress must implement the `score` method that computes a tensor of scores for key-value pairs.
    """

    compression_ratio: float = 0.0

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"

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
        Compute a tensor of fallened scores with shape (bsz, num_key_value_heads * q_len).
        The KV pairs with lowest scores **among all heads in one layer** will be adaptively pruned in the `compress` method.
        """
        raise NotImplementedError





    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The `compress` function adaptively compresses the cache based on scores following the Ada-KV Paradigm. 
        It selects the top-k keys and values among all heads in a layer based on the scores, achieving head-specific compression.

        Example:
            - Batch size (bsz) = 1
            - Number of key-value heads = 2
            - Sequence length (seqlen) = 4
            - Cache budget = 4

        Given:
            (cache) scores = [[head1: [3, 4, 5, 9999], head2: [1, 1, 1, 9998]]]

        The compression process results in:
            compressed (cache) scores = [[head1: [4, 5, 9999], head2: [9998]]]
            flattened (cache) scores = [[4, 5, 9999, 9998]]
        """

        if self.compression_ratio == 0:
            return keys, values

        cache = kwargs.get("past_key_value", None)
        assert cache is not None, "Cache is required for AdaScorerPress"
        cache_metadata = cache.metadata_list[module.layer_idx]

        with torch.no_grad():
            kwargs["metadata"]  = cache_metadata
            flatten_scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        q_len = hidden_states.shape[1]
        num_key_value_heads = cache_metadata.num_key_value_heads

        # Calculate overall budget for one layer
        n_kept = int(q_len * (1 - self.compression_ratio) * num_key_value_heads)

        # NOTE: current implementation only support bsz 1
        assert flatten_scores.shape[0] == 1
        flatten_scores = flatten_scores.view(-1)

        cache_topk_idx = flatten_scores.topk(n_kept, dim=-1).indices
        head_len = cache_metadata.head_lens[0]
        cache_topk_head_idx = cache_topk_idx // head_len

        compressed_head_lens = torch.zeros(num_key_value_heads, dtype=torch.int32,device=keys.device)
        compressed_head_lens.scatter_add_(0, cache_topk_head_idx, torch.ones_like(cache_topk_head_idx, dtype=torch.int32))
        compressed_cu_seqlens_k = torch.cumsum(compressed_head_lens, dim=0, dtype=torch.int32)

        compressed_cu_seqlens_k = torch.cat([torch.tensor([0],dtype=torch.int32,device=keys.device), compressed_cu_seqlens_k])

        compressed_max_seqlen_k = compressed_head_lens.max().cpu().item()
        cache_metadata._update_metadata_while_compressing(compressed_head_lens,compressed_cu_seqlens_k,compressed_max_seqlen_k)

        # sort the cache topk idx, index the retained cache among all heads
        sorted_4_cache_topk_idx = torch.argsort(cache_topk_head_idx,descending=False)
        cache_topk_idx = cache_topk_idx[sorted_4_cache_topk_idx]
        cache_topk_idx = cache_topk_idx.unsqueeze(-1).expand(-1,module.head_dim)
        keys = keys.gather(0, cache_topk_idx).contiguous()
        values = values.gather(0, cache_topk_idx).contiguous()

        return keys, values
