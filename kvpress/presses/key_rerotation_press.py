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
    This function wraps the forward hook of the press object.
    See FINCH: Prompt-guided Key-Value Cache Compression for Large Language Models
    https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280
    for more details on this method.
    Parameters
    ----------
    press : BasePress
        The press object to apply per-layer compression to.
    Returns
    -------
    BasePress
        The press object with rerotation applied.
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

        # Apply RoPE to the pruned keys
        cos, sin = get_position_embeddings(module, keys)
        keys = keys.gather(2, indices).contiguous()

        # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
        rerotation_cos, rerotation_sin = self.get_rerotation_cos_sin(keys, cos, sin)
        keys = (keys * rerotation_cos.unsqueeze(1)) + (rotate_half(keys) * rerotation_sin.unsqueeze(1))

        values = values.gather(2, indices).contiguous()
        return keys, values

    def get_rerotation_cos_sin(self, keys, cos, sin):
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

        # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
        original_cos = cos[:, keys.shape[-2] :]
        shifted_cos = cos[:, -keys.shape[-2]]
        original_sin = sin[:, keys.shape[-2] :]
        shifted_sin = sin[:, -keys.shape[-2]]
        rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
        rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin
        return rerotation_cos, rerotation_sin


def get_position_embeddings(module, x):
    length = x.shape[2]
    # rotary_emb function only needs .device and .dtype, so we can plug in any tensor regardless of shape
    if "position_ids" in inspect.signature(module.rotary_emb.forward).parameters:
        position_ids = torch.arange(length).unsqueeze(0).to(x.device)
        cos, sin = module.rotary_emb(x, position_ids)
    else:
        cos, sin = module.rotary_emb(x, length)
    return cos, sin
