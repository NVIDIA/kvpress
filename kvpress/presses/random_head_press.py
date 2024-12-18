from dataclasses import dataclass

import torch

from kvpress.presses.base_press import BasePress
from kvpress.cache_utils import DynamicHeadCache


@dataclass
class RandomHeadPress(BasePress):
    """
    Randomly masks KV pairs in the cache but with a compression ratio that might differ
    from one head to another.
    """

    compression_ratio: float = 0.0

    def forward_hook(self, module, input, kwargs: dict, output):

        cache = output[-1]
        if cache._seen_tokens > kwargs["hidden_states"].shape[1]:
            return output

        if module.layer_idx == 0:
            # Reset cache indices
            assert isinstance(cache, DynamicHeadCache), "RandomHeadPress requires a DynamicHeadCache"
            assert len(kwargs["hidden_states"]) == 1, "Only batch size 1 is supported"
            cache.indices = []

        random_values = torch.rand_like(cache.key_cache[module.layer_idx][..., 0])
        mask = random_values < torch.quantile(random_values.float(), self.compression_ratio)
        cache.indices.append(torch.nonzero(mask, as_tuple=True))

        return output
