from typing import Optional, Any

import torch
from transformers import DynamicCache

LARGE_NEGATIVE_FLOAT = -float(1e5)


class DynamicHeadCache(DynamicCache):
    """
    Updates the keys and values of the cache at specific indices to ensure that the attention scores are 0 for those
    """

    def __init__(self):
        super().__init__()
        self._seen_tokens = 0
        self.indices = []  # list of indices to mask

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if len(self.indices) > layer_idx:
            # Load query states q from cache_kwargs
            q = cache_kwargs["query_states"]
            bsz, num_heads, seq_len, head_dim = q.shape
            num_key_values_heads = self.key_cache[layer_idx].shape[1]
            num_groups = num_heads // num_key_values_heads

            # Build a fake key k per key group such that for every query q, exp(<q, k>) = 0
            # To do so, use the least square method to find k such that q @ k = LARGE_NEGATIVE_FLOAT
            q = q.view(bsz, num_groups, num_key_values_heads, seq_len, head_dim)
            q = q.transpose(1, 2).reshape(bsz * num_key_values_heads, num_groups * seq_len, head_dim)
            targets = LARGE_NEGATIVE_FLOAT * torch.ones((bsz * num_key_values_heads, seq_len * num_groups)).to(q.device)
            k = torch.linalg.lstsq(q.float(), targets)[0].to(q.dtype)
            assert torch.exp(torch.einsum("hnd,hd->hn", q, k).max()) == 0, "Could not find fake keys"
            k = k.view(bsz, num_key_values_heads, head_dim)

            # At indices, update the keys to the fake keys and the values to 0
            indices = self.indices[layer_idx]
            self.key_cache[layer_idx][*indices] = k[*indices[:2]]
            self.value_cache[layer_idx][*indices] = 0

        return super().update(key_states, value_states, layer_idx, cache_kwargs)
