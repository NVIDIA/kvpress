# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field

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
    with_fix: bool = False

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress)
    
    @staticmethod
    def _rerotate_cos_sin(x, inv_freq, important_pos_batch):
        B, H, L = important_pos_batch.shape
        device = important_pos_batch.device
        device_type = x.device.type
        dtype = x.dtype
        idx = torch.arange(0, L, device=device)
        idx = idx.unsqueeze(0)
        inv_freq = inv_freq[None, None, :, None].float().expand(B, H, -1, 1)  # (B, H, M, 1)
        idx = idx[:, None, :].float().expand(B, H, L)  # (B, H, L)
        delta_pos = idx - important_pos_batch
        delta_pos = delta_pos.unsqueeze(2)  # (B, H, 1, L)

        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = delta_pos.float() * inv_freq.float()
            freqs = freqs.transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().contiguous()
            sin = emb.sin().contiguous()
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

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
        indices = torch.sort(indices, dim=2).values
        if not self.with_fix:
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

            cos, sin = kwargs["position_embeddings"]
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
            cos, sin = cos[:, :n_kept], sin[:, :n_kept]
            keys = (keys * cos.unsqueeze(1)) + (rotate_half(keys) * sin.unsqueeze(1))
        else:
            new_cos, new_sin = self._rerotate_cos_sin(keys, module.rotary_emb.inv_freq, indices)
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
            keys = keys.gather(2, indices).contiguous()
            keys = (keys * new_cos) + (rotate_half(keys) * new_sin)
            
        values = values.gather(2, indices).contiguous()
        return keys, values
