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

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress)
    
    @staticmethod
    def _rerotate_cos_sin(x, inv_freq,  selected_positions):
        """
        Compute cosine and sine rotary positional embeddings required to
        re-rotate pruned keys back into the canonical RoPE space.

        Parameters
        ----------
        x : torch.Tensor
            Key tensor that provides dtype and device information. Shape
            ``(B, H, L, D)``, where *B* is the batch size, *H* is the number
            of attention heads, *L* is the sequence length, and *D* equals
            ``module.head_dim``.
        inv_freq : torch.Tensor
            Inverse-frequency tensor from the layer's ``RotaryEmbedding``
            (``module.rotary_emb.inv_freq``). Shape ``(M,)`` where
            ``M = D // 2``.
         selected_positions : torch.Tensor
            Tensor of kept token indices produced by the pruning step.
            Shape ``(B, H, L)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``cos``, ``sin`` — The cosine and sine embedding tensors, each of
            shape ``(B, H, L, D)`` and matching the ``dtype`` and ``device`` of
            *x*. These tensors are broadcast-multiplied with the gathered keys
            to restore their rotary orientation.
        """
        B, H, L =  selected_positions.shape
        device =  selected_positions.device
        device_type = x.device.type
        dtype = x.dtype
        # Original positional indices
        idx = torch.arange(0, L, device=device)
        idx = idx.unsqueeze(0)
        inv_freq = inv_freq[None, None, :, None].float().expand(B, H, -1, 1)  # (B, H, M, 1)
        idx = idx[:, None, :].float().expand(B, H, L)  # (B, H, L)
        # Compute delta between original and selected positions
        delta_pos = idx -  selected_positions
        delta_pos = delta_pos.unsqueeze(2)  # (B, H, 1, L)

        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            # Compute the new freq by scaling inv_freq by delta
            freqs = delta_pos.float() * inv_freq.float()
            freqs = freqs.transpose(2, 3)
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
            The rerotated keys tensor.
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
