# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from contextvars import ContextVar
from functools import wraps

import torch
from transformers import Gemma3TextModel, LlamaModel, MistralModel, Phi3Model, Qwen2Model, Qwen3Model
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from kvpress.presses.base_press import SUPPORTED_MODELS

_mask_validation_cache: ContextVar[dict | None] = ContextVar("kvpress_mask_validation_cache", default=None)


def _with_mask_validation_cache(func):
    """Keep mask validation local to one text-model forward, releasing it even on failure."""
    if getattr(func, "_kvpress_validation_scope", False):
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Create an isolated validation cache for the model's attention layers."""
        token = _mask_validation_cache.set({})
        try:
            return func(*args, **kwargs)
        finally:
            _mask_validation_cache.reset(token)

    setattr(wrapper, "_kvpress_validation_scope", True)
    return wrapper


def search_hyperplane(X, max_iter: int = 1000):
    """
    Given a tensor X of shape (bsz, seq_len, head_dim), search for a hyperplane Y (bsz, head_dim)
    such that for every i, <X[:, i], Y> <= 0. Returns -1e5 * Y / ||Y||², uniformly scaled down when
    necessary to keep it representable in the input dtype.

    Parameters
    ----------
    X : torch.Tensor
        Query tensor with shape (batch_size, seq_len, head_dim) representing
        the query vectors for which we want to find a nullifying hyperplane.
    max_iter : int, default=1000
        Maximum number of iterations to search for the hyperplane. If no valid
        hyperplane is found within this limit, a ValueError is raised.

    Returns
    -------
    torch.Tensor
        Hyperplane tensor with shape (batch_size, head_dim) and the same dtype as X.

    Raises
    ------
    ValueError
        If no valid hyperplane is found within max_iter iterations.
    """
    output_dtype = X.dtype
    # float16 can overflow during the search and while constructing the fake key.
    if output_dtype == torch.float16:
        X = X.float()

    Y = X.mean(1)  # this initialization is enough for most cases
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        if not mask.any():
            K = -1e5 * Y / Y.norm(dim=-1, keepdim=True) ** 2
            if output_dtype == torch.float16:
                scale = (torch.finfo(output_dtype).max / K.abs().amax(dim=-1, keepdim=True)).clamp(max=1)
                K = (K * scale).to(output_dtype)
            return K
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError("Could not find fake keys such that for every query q, exp(<q, k>) = 0")


def _validate_causal_attention_mask(attention_mask: torch.Tensor, query_length: int) -> None:
    """Check that every batch and head uses an unpadded, unbiased causal mask."""
    error = (
        "Cannot resize this attention mask for a different layer cache. "
        "Only unpadded causal masks can be resized; provide a layer-specific attention mask."
    )
    mask_length = attention_mask.shape[-1]
    if attention_mask.shape[-2] != query_length or mask_length < query_length:
        raise ValueError(error)

    key_positions = torch.arange(mask_length, device=attention_mask.device)
    query_positions = torch.arange(query_length, device=attention_mask.device).unsqueeze(-1)
    allowed = key_positions <= query_positions + mask_length - query_length
    if attention_mask.dtype == torch.bool:
        valid = attention_mask == allowed
    elif attention_mask.dtype.is_floating_point:
        blocked = torch.isneginf(attention_mask) | (attention_mask == torch.finfo(attention_mask.dtype).min)
        valid = torch.where(allowed, attention_mask == 0, blocked)
    else:
        raise ValueError(error)
    if not valid.all():
        raise ValueError(error)


def _validate_shared_causal_attention_mask(attention_mask: torch.Tensor, query_length: int) -> None:
    """Reuse successful validation of an unchanged shared mask within the current forward."""
    cache = _mask_validation_cache.get()
    if cache is not None:
        version = None if attention_mask.is_inference() else attention_mask._version
        cache_key = (id(attention_mask), version, query_length)
        if cache.get(cache_key) is attention_mask:
            return
    _validate_causal_attention_mask(attention_mask, query_length)
    if cache is not None:
        cache[cache_key] = attention_mask


def _pad_causal_attention_mask(attention_mask: torch.Tensor, query_length: int, padding_length: int) -> torch.Tensor:
    """Validate an unpadded causal mask and prepend visible past columns when needed."""
    _validate_shared_causal_attention_mask(attention_mask, query_length)
    if padding_length == 0:
        return attention_mask
    visible = True if attention_mask.dtype == torch.bool else 0.0
    return torch.nn.functional.pad(attention_mask, (padding_length, 0), value=visible)


def attention_patch(func, *, is_sdpa: bool = False):
    """
    Decorator to update the keys before the attention computation at the indices provided in module.masked_key_indices
    The keys are updated with a fake key k whose scaled attention weight underflows to zero
    This solution is not optimal as it does not reduce peak memory and slightly increases runtime

    Parameters
    ----------
    func : callable
        The original attention function to be patched. Should accept parameters
        (module, query, key, value, attention_mask, dropout, **kwargs).
    is_sdpa : bool, default=False
        Whether an omitted mask needs SDPA's causal alignment corrected for longer caches.

    Returns
    -------
    callable
        The wrapped attention function that supports head-wise key masking.
    """

    @wraps(func)
    def wrapper(module, query, key, value, attention_mask, dropout=0.0, **kwargs):
        """Apply head-wise key masking and align validated causal masks to the layer cache."""
        if query.shape[2] == key.shape[2]:
            # Prefilling
            module.masked_key_indices = None
        elif getattr(module, "masked_key_indices", None) is not None and module.masked_key_indices[-1].numel() > 0:
            # Decoding: build fake keys k s.t. exp(<q, k>) = 0
            bsz, num_heads, seq_len, head_dim = query.shape
            num_key_value_heads = key.shape[1]
            num_groups = num_heads // num_key_value_heads

            # Build a fake key k per key group such that for every query q, exp(<q, k>) = 0
            q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
            q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
            k = search_hyperplane(q)
            k = k.view(bsz, num_key_value_heads, head_dim)

            # At indices, update the keys to the fake keys
            batch_indices, head_indices, seq_indices = module.masked_key_indices
            key[batch_indices, head_indices, seq_indices] = k[batch_indices, head_indices]

        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4 and attention_mask.shape[-1] != 1:
            # The shared mask uses the first layer's cache length. New query tokens
            # are at the right edge of every layer, regardless of its context budget.
            difference = key.shape[-2] - attention_mask.shape[-1]
            if difference < 0:
                _validate_shared_causal_attention_mask(attention_mask, query.shape[-2])
                attention_mask = attention_mask[..., -key.shape[-2] :]
            elif difference > 0:
                if kwargs.get("sliding_window") is not None or getattr(module, "sliding_window", None) is not None:
                    raise ValueError(
                        "Cannot extend a sliding-window attention mask for a longer layer cache; "
                        "provide a layer-specific attention mask."
                    )
                attention_mask = _pad_causal_attention_mask(attention_mask, query.shape[-2], difference)

        is_causal = kwargs.get("is_causal")
        if is_causal is None:
            is_causal = getattr(module, "is_causal", True)
        if is_sdpa and attention_mask is None and 1 < query.shape[-2] < key.shape[-2] and is_causal:
            # An empty first layer lets Transformers omit the shared mask, but
            # later layers still need a bottom-right-aligned causal mask.
            if kwargs.get("sliding_window") is not None or getattr(module, "sliding_window", None) is not None:
                raise ValueError(
                    "Cannot infer a sliding-window mask for a longer layer cache; "
                    "provide a layer-specific attention mask."
                )
            key_positions = torch.arange(key.shape[-2], device=query.device)
            query_positions = torch.arange(query.shape[-2], device=query.device).unsqueeze(-1)
            attention_mask = (key_positions <= query_positions + key.shape[-2] - query.shape[-2])[None, None]

        # see https://github.com/NVIDIA/kvpress/pull/115#issuecomment-3183785597
        # cu_seq_lens_k are only in kwargs if model.generate is used.
        if "cu_seq_lens_k" in kwargs:
            kwargs["cu_seq_lens_k"][-1] = key.shape[-2]
        return func(module, query, key, value, attention_mask, dropout=dropout, **kwargs)

    return wrapper


def patch_attention_functions():
    """
    Apply attention patching to registered backends and supported models' eager functions.

    This function automatically patches all attention functions registered in
    transformers' ALL_ATTENTION_FUNCTIONS to support head-wise key masking.
    It also patches the model-specific eager functions, which are not registered there.
    This enables head-specific masking (like AdaKV) and unequal per-layer caches
    (like PyramidKV) to work correctly during text generation.

    The patching is applied globally and affects all transformer models loaded
    after this function is called. It's automatically called when importing
    kvpress to ensure compatibility with head-wise compression methods.

    Notes
    -----
    This function modifies the global attention functions in the transformers library.
    Attention masks with the correct key length, singleton key dimensions, and
    two-dimensional padding masks are left unchanged. Unpadded causal masks are
    right-aligned to each layer's keys. Longer caches can extend them with visible
    past-context columns. Restricted masks, including padded and sliding-window
    masks, require a layer-specific mask when the key length differs.
    Successful mask validation is shared across the supported text model's layers
    for one forward pass only. If SDPA omits the shared mask, longer layers with
    multiple query tokens receive an explicit causal mask aligned to their keys.
    """
    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        ALL_ATTENTION_FUNCTIONS[name] = attention_patch(func, is_sdpa=name == "sdpa")
    for model in SUPPORTED_MODELS:
        modeling_module = importlib.import_module(model.__module__)
        eager_attention = getattr(modeling_module, "eager_attention_forward")
        setattr(modeling_module, "eager_attention_forward", attention_patch(eager_attention))
    for text_model in (LlamaModel, MistralModel, Phi3Model, Qwen2Model, Qwen3Model, Gemma3TextModel):
        setattr(text_model, "forward", _with_mask_validation_cache(text_model.forward))
