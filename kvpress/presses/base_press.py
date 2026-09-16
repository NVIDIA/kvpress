# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Generator, Optional

import torch
from torch import nn
from transformers import Gemma3ForConditionalGeneration, PreTrainedModel

from kvpress.adapters import get_adapter, get_adapter_from_module, has_adapter

logger = logging.getLogger(__name__)


def is_prefilling(cache_position: torch.Tensor, q_len: int) -> bool:
    """Return whether the current forward pass is the initial prefill."""
    prefilling = cache_position[-1] + 1 == q_len
    return bool(prefilling.item() if isinstance(prefilling, torch.Tensor) else prefilling)


@dataclass
class BasePress:
    """
    Base class for all KV cache compression methods.

    This class provides the foundation for implementing various key-value cache compression
    techniques. Subclasses must implement the `compress` method to define their specific
    compression logic.

    The compression is applied only during pre-filling (not during generation).
    """

    def post_init_from_model(self, model: PreTrainedModel):
        """
        Optional method to initialize press parameters from the model
        """
        pass

    def warn_unsupported_model(self, model: PreTrainedModel) -> None:
        """Warn when no adapter is registered for this architecture."""
        if not has_adapter(model):
            logger.warning(
                f"No KVPress adapter registered for model_type of {type(model).__name__}; "
                "falling back to the Llama-like adapter, which is untested for this architecture."
            )

    @contextmanager
    def hook_scope(self, model: PreTrainedModel, hook: Optional[Callable] = None) -> Generator:
        """Install this press's forward hooks for the duration of the block.

        The adapter decides *which* modules get hooked; presses differ only in how long
        the hooks stay installed, which is what this scope expresses. Presses overriding
        ``__call__`` should use this instead of registering hooks themselves.
        """
        hooks = get_adapter(model).register_forward_hooks(model, hook or self.forward_hook)
        try:
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()

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
        The core logic of the compression method.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer where compression is applied.
        hidden_states : torch.Tensor
            Hidden states of the current layer with shape (batch_size, seq_len, hidden_dim).
            These represent the input to the attention layer.
        keys : torch.Tensor
            Key tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
            These are keys ready for compression.
        values : torch.Tensor
            Value tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
            These are values ready for compression.
        attentions : torch.Tensor
            Attention weights from the layer with shape (batch_size, num_heads, seq_len, seq_len).
            May be None if attention weights are not computed or needed.
        kwargs : dict
            Additional keyword arguments from the forward pass.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the compressed keys and values tensors. The returned tensors
            should have reduced sequence length dimension compared to the input tensors.
        """

        raise NotImplementedError("compress method must be implemented in subclass")

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Default forward hook called after the forward pass of an attention layer.

        This hook automatically applies compression during the pre-filling phase by:
        1. Checking if we're still in pre-filling (not generation) phase
        2. Extracting keys and values from the cache (handling quantization)
        3. Calling the compress method to reduce the cache size
        4. Updating the cache with compressed keys and values

        The hook ensures compression is only applied during pre-filling and correctly
        handles both quantized and unquantized caches.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer.
        input : list[torch.Tensor]
            Input tensors to the forward pass of the attention layer. This parameter
            is provided by PyTorch's hook mechanism but not used in the default implementation.
        kwargs : dict
            Keyword arguments passed to the attention layer's forward method, including:
            - hidden_states: Input embeddings to the attention layer
            - past_key_values: The KV cache object being modified
            - cache_position: Position indices indicating where we are in the sequence
            - position_embeddings: RoPE embeddings if applicable
        output : list
            Output from the attention layer's forward pass. Contains:
            - [0]: Hidden states output
            - [1]: Attention weights (may be None)

        Returns
        -------
        list
            The potentially modified output from the forward pass. This
            is the same as the input output, but the underlying cache has been compressed in-place.
        """

        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]

        # Don't compress after pre-filling
        if not is_prefilling(kwargs["cache_position"], q_len):
            return output

        adapter = get_adapter_from_module(module)
        keys, values = adapter.get_keys_values(cache, module)
        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)
        adapter.set_keys_values(cache, module, keys, values)

        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager to apply a compression method to a model.

        This method registers forward hooks on all attention layers of the model to enable
        automatic KV cache compression during the pre-filling phase. The hooks are automatically
        removed when exiting the context manager.

        Apply this context manager during the pre-filling phase to compress the context.

        Parameters
        ----------
        model : PreTrainedModel
            The transformer model to apply compression to.

        Examples
        --------
        >>> from kvpress import KnormPress
        >>> press = KnormPress(compression_ratio=0.5)
        >>> with press(model):
        ...     # Forward pass with compression applied
        ...     outputs = model(input_ids, past_key_values=cache)
        """
        self.warn_unsupported_model(model)

        if isinstance(model, Gemma3ForConditionalGeneration):
            logger.warning_once("Compression in Gemma3 is only applied to layer without sliding window attention")

        self.post_init_from_model(model)
        with self.hook_scope(model):
            yield
