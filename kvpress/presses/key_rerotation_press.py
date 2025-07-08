# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class KeyRerotationPress(BasePress):
    """
    Key Rerotation: RoPE-aware compression wrapper for maintaining positional encoding.

    Enhances any ScorerPress by applying key rerotation after compression to maintain
    proper RoPE (Rotary Position Embedding) representations. When tokens are pruned,
    remaining tokens need positional encodings adjusted for their new positions.
    This method is used in several key-value cache compression methods, such as
    - SinkCache implementation in Hugging Face's transformers library
    - FINCH: Prompt-guided Key-Value Cache Compression for Large Language Models

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method to enhance with key rerotation.
        Rerotation is applied after the press determines which tokens to keep.
    """

    press: ScorerPress

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    @staticmethod
    def _rerotate_cos_sin(x, inv_freq, selected_positions):
        """
        Compute cosine and sine rotary positional embeddings required to
        re-rotate pruned keys back into the canonical RoPE space.

        Parameters
        ----------
        x : torch.Tensor
            Any key-like tensor that provides ``dtype`` and ``device``.
            Shape ``(bsz, num_key_value_heads, q_len, d)``.
        inv_freq : torch.Tensor
            ``module.rotary_emb.inv_freq``. Shape ``(d//2,)``.
        selected_positions : torch.Tensor
            Indices of the *kept* tokens.
            Shape ``(bsz, num_key_value_heads, n_kept)``.

        Returns
        -------
        cos, sin : torch.Tensor
            Cosine and sine embeddings, each of shape
            ``(bsz, num_key_value_heads, n_kept, d)``, matching ``dtype``/``device`` of ``x``.
        """
        bsz, num_key_value_heads, n_kept = selected_positions.shape
        device = selected_positions.device
        device_type = x.device.type
        dtype = x.dtype
        # Original positional indices
        idx = torch.arange(0, n_kept, device=device)  # (n_kept,)
        idx = idx.unsqueeze(0)  # (1, n_kept)
        inv_freq = inv_freq[None, None, :, None].float().expand(bsz, num_key_value_heads, -1, 1)
        idx = idx[:, None, :].float().expand(bsz, num_key_value_heads, n_kept)
        # Compute delta between original and selected positions
        delta_pos = idx - selected_positions  # (bsz, num_key_value_heads, n_kept)
        delta_pos = delta_pos.unsqueeze(2)  # (bsz, num_key_value_heads, 1, n_kept)

        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            # Compute the new freq by scaling inv_freq by delta
            freqs = delta_pos.float() * inv_freq.float()  # (bsz, num_key_value_heads, d//2, n_kept)
            freqs = freqs.transpose(2, 3)  # (bsz, num_key_value_heads, n_kept, d//2)
            emb = torch.cat((freqs, freqs), dim=-1)
            # Compute cosine and sine required to re-rotate keys to selected positions
            cos = emb.cos().contiguous()
            sin = emb.sin().contiguous()
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    @staticmethod
    def rerotate_keys(
        module: nn.Module,
        indices: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rerotate keys to have a uniform RoPE representation of keys after pruning.

        Parameters
        ----------
        module : nn.Module
            The model module containing the rotary embedding.
        indices : torch.Tensor
            Indices of the kept tokens after pruning.
        keys : torch.Tensor
            The keys tensor to be rerotated.

        Returns
        -------
        torch.Tensor
            The rerotated keys tensor of shape
            ``(bsz, num_heads, n_kept, d)``.
        """
        new_cos, new_sin = KeyRerotationPress._rerotate_cos_sin(keys, module.rotary_emb.inv_freq, indices)
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()
        return (keys * new_cos) + (rotate_half(keys) * new_sin)

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
        keys = self.rerotate_keys(module, indices, keys)
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        values = values.gather(2, indices).contiguous()
        return keys, values
