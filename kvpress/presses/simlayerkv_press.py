import logging
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.snapkv_press import SnapKVPress

logger = logging.getLogger(__name__)


@dataclass
class SimLayerKVPress(BasePress):
    """
    SimLayerKV (https://arxiv.org/abs/2410.13846) uses a layer-wise approach to compression:
        - layers identified as lazy use the Streaming LLM approach (only initial and recent KV pairs are kept)
        - other layers use the full KV cache

    To identify lazy layers, the last attention weights are used. If the sum of attention weights of the last tokens
    over the initial and recent tokens is above the lazy_threshold, the layer is considered lazy.

    Recommended values for lazy_threshold from the official repository:
        - llama3: 0.9
        - llama2: 0.65
        - mistral: 0.8
        - qwen: 0.85
    By default, lazy_threshold is set to 1.0 (no compression)
    (Source: https://github.com/sail-sg/SimLayerKV/blob/main/LongBench/pred.py#L167)
    """

    lazy_threshold: float = 1.0
    n_last: int = 1  # n_last=1 to match SKLV-decode
    n_recent: int = 1024
    n_initial: int = 4

    def __post_init__(self):
        assert 0.0 <= self.lazy_threshold <= 1.0, "lazy_threshold should be in [0, 1]"
        self.compression_ratios = []

    def is_lazy(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> bool:
        """
        Compute the attention weights of the last tokens over the initial and recent tokens.
        The layer is considered lazy if the sum of these attention weights is above the lazy_threshold.
        """

        attn_weights = SnapKVPress.compute_window_attention(
            module, hidden_states, keys, self.n_last, position_embeddings
        )
        attn_weights = attn_weights.mean((0, 1, 2))  # mean over bsz, heads and window size
        score = attn_weights[: self.n_initial].sum() + attn_weights[-self.n_recent :].sum()
        return score.item() > self.lazy_threshold

    @property
    def compression_ratio(self):
        if len(self.compression_ratios) > 0:
            return sum(self.compression_ratios) / len(self.compression_ratios)
        else:
            raise ValueError("Forward pass must be run to compute the compression ratio")

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Initialize the compression ratios
        if module.layer_idx == 0:
            self.compression_ratios = []

        # Check if compression is needed
        q_len = hidden_states.shape[1]
        min_length = self.n_initial + self.n_recent + self.n_last

        if q_len <= min_length:
            logger.warning(f"Sequence length is shorter than {min_length}: no compression applied")

        if (self.lazy_threshold == 1.0) or (q_len <= min_length):
            self.compression_ratios.append(0.0)
            return keys, values

        # Compression
        if self.is_lazy(module, hidden_states, keys, kwargs["position_embeddings"]):
            # If layer is lazy, only keep the initial and recent KV pairs
            keys = torch.cat([keys[:, :, : self.n_initial], keys[:, :, -self.n_recent + self.n_last :]], dim=2)
            values = torch.cat([values[:, :, : self.n_initial], values[:, :, -self.n_recent + self.n_last :]], dim=2)
            self.compression_ratios.append((q_len - self.n_initial - self.n_recent + 1) / q_len)
        else:
            self.compression_ratios.append(0.0)

        return keys, values
