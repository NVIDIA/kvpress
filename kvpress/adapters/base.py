# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture adapters for KV cache access, hook install, and Q/K projection.

Presses should depend on this interface rather than on model-class ``isinstance`` checks.
Register a new adapter with ``register_adapter("model_type")``; unknown types fall back
to the default Llama-like adapter.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn
from transformers import PreTrainedModel

# Not every architecture caches through ``transformers.Cache``: hybrid stacks such as
# Qwen3.5 use their own container that also holds recurrent state.
CacheLike = Any

_ADAPTER_BY_MODEL_TYPE: dict[str, type["ModelAdapter"]] = {}
_DEFAULT_ADAPTER_CLS: Optional[type["ModelAdapter"]] = None


def register_adapter(*model_types: str, default: bool = False):
    """Register an adapter for one or more ``config.model_type`` strings."""

    def decorator(cls: type["ModelAdapter"]) -> type["ModelAdapter"]:
        for model_type in model_types:
            _ADAPTER_BY_MODEL_TYPE[model_type] = cls
        if default:
            global _DEFAULT_ADAPTER_CLS
            _DEFAULT_ADAPTER_CLS = cls
        return cls

    return decorator


def _model_type_from_config(config) -> Optional[str]:
    if config is None:
        return None
    model_type = getattr(config, "model_type", None)
    if model_type:
        return model_type
    text_config = getattr(config, "text_config", None)
    return getattr(text_config, "model_type", None) if text_config is not None else None


def get_adapter(model: PreTrainedModel) -> "ModelAdapter":
    """Return the adapter for ``model``; Llama-like is the default fallback."""
    return get_adapter_from_model_type(_model_type_from_config(getattr(model, "config", None)))


def get_adapter_from_module(module: nn.Module) -> "ModelAdapter":
    """Return the adapter using the attention module's config (available in forward hooks)."""
    return get_adapter_from_model_type(_model_type_from_config(getattr(module, "config", None)))


def has_adapter(model: PreTrainedModel) -> bool:
    """Whether an adapter is explicitly registered for ``model`` (rather than falling back)."""
    model_type = _model_type_from_config(getattr(model, "config", None))
    return (model_type or "") in _ADAPTER_BY_MODEL_TYPE


def get_adapter_from_model_type(model_type: Optional[str]) -> "ModelAdapter":
    cls = _ADAPTER_BY_MODEL_TYPE.get(model_type or "")
    if cls is None:
        if _DEFAULT_ADAPTER_CLS is None:
            raise RuntimeError("No default KVPress model adapter is registered")
        cls = _DEFAULT_ADAPTER_CLS
    return cls()


class ModelAdapter:
    """Architecture-specific KV cache and attention-module helpers."""

    def language_model(self, model: PreTrainedModel) -> nn.Module:
        raise NotImplementedError

    def compressible_modules(self, model: PreTrainedModel) -> list[nn.Module]:
        raise NotImplementedError

    def prepare_module(self, model: PreTrainedModel, module: nn.Module) -> None:
        """Called once before a hook is registered (e.g. attach ``rotary_emb``)."""

    def register_forward_hooks(self, model: PreTrainedModel, hook: Callable) -> list:
        hooks = []
        for module in self.compressible_modules(model):
            self.prepare_module(model, module)
            hooks.append(module.register_forward_hook(hook, with_kwargs=True))
        return hooks

    def get_keys_values(self, cache: CacheLike, module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def set_keys_values(
        self, cache: CacheLike, module: nn.Module, keys: torch.Tensor, values: torch.Tensor
    ) -> None:
        raise NotImplementedError

    def make_cache(self, model: PreTrainedModel) -> CacheLike:
        raise NotImplementedError

    def supports_multi_token_continuation(self) -> bool:
        """Whether a populated cache can be extended by more than one token per forward pass.

        Architectures with recurrent state may only be able to carry that state forward
        one token at a time, in which case callers must feed follow-up tokens singly.
        """
        return True

    def rewind_cache(self, cache: CacheLike, seq_lengths: list[int]) -> None:
        raise NotImplementedError

    def prerope_queries(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prerope_keys(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_rope(
        self, module: nn.Module, states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Rotate ``states`` of shape ``(bsz, n_heads, seq_len, head_dim)`` with the given cos/sin.

        Presses that recompute queries must go through this rather than assuming a
        Llama-style full rotation, since some architectures rotate only part of the
        head dimension.
        """
        raise NotImplementedError
