# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass

import torch
from torch import nn
from transformers import QuantizedCache

from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)


@dataclass
class ScorerPress(BasePress):
    """
    Default press method for using a score method.
    The `forward_hook` method is called after the forward pass of an attention layer.
    and updates the cache with the pruned KV pairs.
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
        """Compute the scores for each KV pair in the layer.

        Parameters
        ----------
        module :
            Transformer layer, see `hook` method for more details.
        hidden_states :
            Hidden states of the layer.
        keys :
            Keys of the cache. Note keys are after RoPE.
        values :
            Values of the cache.
        attentions :
            Attention weights of the layer.
        kwargs :
            Keyword arguments, as given to the forward pass of the layer.

        Returns
        -------
            Scores for each KV pair in the layer, shape keys.shape[:-1].
        """
        raise NotImplementedError

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Default cache compression hook called after the forward pass of an attention layer.
        The hook is applied only during the pre-filling phase if there is some pruning ratio.
        This implementation allows to remove a constant number of KV pairs.

        Parameters
        ----------
        module :
            Transformer attention layer.
        input :
            Input to the hook. This is the input to the forward pass of the layer.
        kwargs :
            Keyword arguments, as given to the forward pass of the layer.
        output :
            Output of the hook. This is the original output of the forward pass of the layer.

        Returns
        -------
            Modified output of the forward pass of the layer.

        """
        # See e.g. LlamaDecoderLayer.forward for the output structure
        if len(output) == 3:
            _, attentions, cache = output
        else:
            attentions, cache = None, output[-1]

        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]

        # Don't compress if the compression ratio is 0 or this is not pre-filling
        if (self.compression_ratio == 0) or (cache.seen_tokens > q_len):
            return output

        if isinstance(cache, QuantizedCache):
            keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])
            values = cache._dequantize(cache._quantized_value_cache[module.layer_idx])
        else:
            keys = cache.key_cache[module.layer_idx]
            values = cache.value_cache[module.layer_idx]

        with torch.no_grad():
            scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Prune KV pairs with the lowest scores
        n_kept = int(q_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Update cache
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        if isinstance(cache, QuantizedCache):
            cache._quantized_key_cache[module.layer_idx] = cache._quantize(keys, axis=cache.axis_key)
            cache._quantized_value_cache[module.layer_idx] = cache._quantize(values, axis=cache.axis_value)
        else:
            cache.key_cache[module.layer_idx] = keys
            cache.value_cache[module.layer_idx] = values

        return output
