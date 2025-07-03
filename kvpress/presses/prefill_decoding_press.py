# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .base_press import BasePress
from .decoding_press import DecodingPress

logger = logging.getLogger(__name__)


@dataclass
class PrefillDecodingPress(BasePress):
    """
    A wrapper press that combines separate prefilling and decoding compression strategies.
    
    This press acts as a single press interface but internally delegates to different
    presses based on the current phase (prefilling vs decoding). During prefilling,
    it uses the prefilling_press. During decoding, it uses the decoding_press.
    
    Parameters
    ----------
    prefilling_press : BasePress, optional
        Press to use during the prefilling phase. If None, no compression is applied during prefilling.
    decoding_press : DecodingPress, optional
        Press to use during the decoding phase. If None, no compression is applied during decoding.
    """
    
    prefilling_press: Optional[BasePress] = None
    decoding_press: Optional[DecodingPress] = None
    
    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV cache using the appropriate press based on current phase.
        
        Parameters
        ----------
        module : nn.Module
            Transformer attention layer
        hidden_states : torch.Tensor
            Current hidden states
        keys : torch.Tensor
            Keys of the cache (unquantized)
        values : torch.Tensor
            Values of the cache (unquantized)
        attentions : torch.Tensor
            Attention weights of the layer
        kwargs : dict
            Keyword arguments from the forward pass
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated keys and values
        """
        q_len = hidden_states.shape[1]
        
        # Determine if we're in prefilling or decoding phase
        if kwargs["cache_position"][-1] <= q_len:
            # Prefilling phase
            if self.prefilling_press is not None:
                return self.prefilling_press.compress(
                    module, hidden_states, keys, values, attentions, kwargs
                )
        else:
            # Decoding phase
            if self.decoding_press is not None:
                return self.decoding_press.compress(
                    module, hidden_states, keys, values, attentions, kwargs
                )
                
        # No compression applied
        return keys, values
        
    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Forward hook that delegates to the appropriate press based on current phase.
        """
        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]
        
        # Determine if we're in prefilling or decoding phase
        if kwargs["cache_position"][-1] <= q_len:
            # Prefilling phase
            if self.prefilling_press is not None:
                return self.prefilling_press.forward_hook(module, input, kwargs, output)
        else:
            # Decoding phase
            if self.decoding_press is not None:
                return self.decoding_press.forward_hook(module, input, kwargs, output)
                
        # No hook applied
        return output
        
    def __call__(self, model: PreTrainedModel):
        """
        Context manager to apply the combined press to a model.
        
        This uses the base press logic and adds decoding press reset at the end.
        """
        try:
            with super().__call__(model):
                yield
        finally:
            # Reset decoding press if it exists
            if self.decoding_press is not None:
                self.decoding_press.reset()
                
    def reset(self):
        """Reset both presses."""
        if self.decoding_press is not None:
            self.decoding_press.reset()
