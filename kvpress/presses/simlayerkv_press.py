from dataclasses import dataclass

import torch
from torch import nn
from transformers import QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.presses.snapkv_press import SnapKVPress


@dataclass
class SimLayerKVPress(BasePress):
    """
    SimLayerKV (https://arxiv.org/abs/2410.13846) uses a layer-wise approach to compression:
        - layers identified as lazy use the Streaming LLM approach (only initial and recent KV pairs are kept)
        - other layers use the full KV cache

    To identify lazy layers, the last attention weights are used. If the sum of attention weights of the last tokens
    over the initial and recent tokens is above the lazy_threshold, the layer is considered lazy.

    Official implementation: https://github.com/sail-sg/SimLayerKV. We use n_last=1 to match SKLV-decode
    """

    lazy_threshold: float
    n_last: int = 1
    n_recent: int = 1024
    n_initial: int = 4

    def is_lazy(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
    ) -> bool:
        """
        Compute the average attention weights of the last tokens over the initial and recent tokens.
        A slight difference with the original implementation is that we
        """

        attn_weights = SnapKVPress.compute_window_attention(module, hidden_states, keys, self.n_last)
        attn_weights = attn_weights.mean((0, 1, 2))  # mean over bsz, heads and window size
        score = attn_weights[: self.n_initial].sum() + attn_weights[-self.n_recent :].sum()
        return score.item() > self.lazy_threshold

    @property
    def compression_ratio(self):
        if hasattr(self, "compression_ratios"):
            return sum(self.compression_ratios) / len(self.compression_ratios)
        else:
            raise ValueError("Forward pass must be run to compute the compression ratio")

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):

        cache = output[-1]
        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]

        # Don't compress if this is not pre-filling or if there are not enough tokens
        if (cache.seen_tokens > q_len) or (cache.seen_tokens < self.n_initial + self.n_recent):
            return output

        # Re-initialize the compression_ratios list
        if module.layer_idx == 0:
            self.compression_ratios = []
            assert hidden_states.shape[1] > self.n_last, "Query length should be greater than the window size"

        if isinstance(cache, QuantizedCache):
            keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])
            values = cache._dequantize(cache._quantized_value_cache[module.layer_idx])
        else:
            keys = cache.key_cache[module.layer_idx]
            values = cache.value_cache[module.layer_idx]

        if self.is_lazy(module, hidden_states, keys):
            # If layer is lazy, only keep the initial and recent KV pairs
            keys = torch.cat([keys[:, :, : self.n_initial], keys[:, :, -self.n_recent + self.n_last :]], dim=2)
            values = torch.cat([values[:, :, : self.n_initial], values[:, :, -self.n_recent + self.n_last :]], dim=2)
            self.compression_ratios.append((q_len - self.n_initial - self.n_recent + 1) / q_len)
        else:
            self.compression_ratios.append(0)

        if isinstance(cache, QuantizedCache):
            cache._quantized_key_cache[module.layer_idx] = cache._quantize(keys, axis=cache.axis_key)
            cache._quantized_value_cache[module.layer_idx] = cache._quantize(values, axis=cache.axis_value)
        else:
            cache.key_cache[module.layer_idx] = keys
            cache.value_cache[module.layer_idx] = values

        return output
