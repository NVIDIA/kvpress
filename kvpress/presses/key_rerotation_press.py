# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import inspect
from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class KeyRerotationPress(BasePress):
    """
    Rerotate keys to have a uniform RoPE representation of keys after pruning.
    This method is used in several key-value cache compression methods, such as
    - SinkCache implementation in Hugging Face's transformers library
    - FINCH: Prompt-guided Key-Value Cache Compression for Large Language Models
    Parameters
    ----------
    press : ScorerPress
        The press object to apply per-layer compression to.
    """

    press: ScorerPress

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.press.compression_ratio == 0:
            return keys, values

        # Compute scores from base press
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        # Get indices of KV pairs with the lowest scores
        q_len = hidden_states.shape[1]
        n_kept = int(q_len * (1 - self.press.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        cos, sin = get_rope_embeddings(module, keys)
        # Rerotate as follows
        #  1. keys = RoPE(W_k * hidden_states)
        #  2. keys_unrotated = RoPE^-1(keys)
        #  3. keys_pruned = prune(keys_unrotated)
        #  4. keys = RoPE(keys_pruned)

        # 2. Inverse of rotation matrix is equivalent to setting sin -> -sin in the equation below
        keys = (keys * cos.unsqueeze(1)) + (rotate_half(keys) * (-sin.unsqueeze(1)))
        # 3. Prune keys
        keys = keys.gather(2, indices).contiguous()
        # 4. Apply RoPE
        cos, sin = get_rope_embeddings(module, keys)
        keys = (keys * cos.unsqueeze(1)) + (rotate_half(keys) * sin.unsqueeze(1))

        values = values.gather(2, indices).contiguous()
        return keys, values


def get_rope_embeddings(module, x):
    length = x.shape[2]
    # rotary_emb function only needs .device and .dtype, so we can plug in any tensor regardless of shape
    if "position_ids" in inspect.signature(module.rotary_emb.forward).parameters:
        position_ids = torch.arange(length).unsqueeze(0).to(x.device)
        cos, sin = module.rotary_emb(x, position_ids)
    else:
        cos, sin = module.rotary_emb(x, length)
    return cos, sin
