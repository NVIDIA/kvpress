# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def search_hyperplane(X, max_iter: int = 1000, attention_scaling: float = 1.0):
    """
    Given a tensor X of shape (bsz, seq_len, head_dim), return fake keys K (bsz, head_dim)
    such that exp(attention_scaling * <X[:, i], K>) underflows to zero for every i.

    Starting from the mean query direction, repeatedly add queries that are on the wrong side
    of the hyperplane. Once every query has a positive dot product with that direction, negate
    and scale it to produce a fake key with a sufficiently negative attention logit.

    Parameters
    ----------
    X : torch.Tensor
        Query tensor with shape (batch_size, seq_len, head_dim) representing
        the query vectors for which we want to find a nullifying hyperplane.
    max_iter : int, default=1000
        Maximum number of iterations to search for the hyperplane. If no valid
        hyperplane is found within this limit, a ValueError is raised.
    attention_scaling : float, default=1.0
        Scaling applied to query-key products by the attention implementation.

    Returns
    -------
    torch.Tensor
        Fake key tensor with shape (batch_size, head_dim) and the same dtype as X.

    Raises
    ------
    ValueError
        If no valid hyperplane is found or the required fake key is not representable.
    """
    if attention_scaling <= 0:
        raise ValueError("attention_scaling must be positive")

    output_dtype = X.dtype
    # Float16 can overflow during the search, before the fake key can be rescaled.
    if X.dtype == torch.float16:
        X = X.float()

    Y = X.mean(1)  # this initialization is enough for most cases
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        if not mask.any():
            return _build_finite_fake_keys(X, Y, output_dtype, attention_scaling)
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError("Could not find a hyperplane that nullifies every query")


def attention_patch(func):
    """
    Decorator to update the keys before the attention computation at the indices provided in module.masked_key_indices
    The keys are updated with a fake key k whose scaled attention weight underflows to zero
    This solution is not optimal as it does not reduce peak memory and slightly increases runtime

    Parameters
    ----------
    func : callable
        The original attention function to be patched. Should accept parameters
        (module, query, key, value, attention_mask, dropout, **kwargs).

    Returns
    -------
    callable
        The wrapped attention function that supports head-wise key masking.
    """

    def wrapper(module, query, key, value, attention_mask, dropout, **kwargs):
        if query.shape[2] == key.shape[2]:
            # Prefilling
            module.masked_key_indices = None
        elif getattr(module, "masked_key_indices", None) is not None:
            # Decoding: build fake keys whose scaled attention weights underflow to zero.
            bsz, num_heads, seq_len, head_dim = query.shape
            num_key_value_heads = key.shape[1]
            num_groups = num_heads // num_key_value_heads

            # Build one fake key per key group that nullifies every query in that group.
            q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
            q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
            attention_scaling = kwargs.get("scaling")
            if attention_scaling is None:
                attention_scaling = getattr(module, "scaling", head_dim**-0.5)
            k = search_hyperplane(q, attention_scaling=attention_scaling)
            k = k.view(bsz, num_key_value_heads, head_dim)

            # At indices, update the keys to the fake keys
            batch_indices, head_indices, seq_indices = module.masked_key_indices
            key[batch_indices, head_indices, seq_indices] = k[batch_indices, head_indices]

        # see https://github.com/NVIDIA/kvpress/pull/115#issuecomment-3183785597
        # cu_seq_lens_k are only in kwargs if model.generate is used.
        if "cu_seq_lens_k" in kwargs:
            kwargs["cu_seq_lens_k"][-1] = key.shape[-2]
        return func(module, query, key, value, attention_mask, dropout, **kwargs)

    return wrapper


def patch_attention_functions():
    """
    Apply attention patching to all transformer attention functions.

    This function automatically patches all attention functions registered in
    transformers' ALL_ATTENTION_FUNCTIONS to support head-wise key masking.
    It enables KVPress compression methods that require head-specific masking
    (like AdaKV) to work correctly during text generation.

    The patching is applied globally and affects all transformer models loaded
    after this function is called. It's automatically called when importing
    kvpress to ensure compatibility with head-wise compression methods.

    Notes
    -----
    This function modifies the global attention functions in the transformers
    library. The modifications do not affect models that don't use head-wise compression (i.e. don't have
    module.masked_key_indices).
    """
    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        ALL_ATTENTION_FUNCTIONS[name] = attention_patch(func)


def _compute_safe_margins(X, Y, output_dtype):
    """
    Return a conservative lower bound for every positive query-hyperplane margin.

    A margin is ``q·Y``: it measures how strongly a query lies on the positive side of
    the separating hyperplane. Floating-point rounding can make the measured margin
    slightly larger than the effective margin used by attention. If that overestimate
    were used directly, the fake key could be scaled too weakly.

    This function bounds the rounding from four operations:
    1. measuring ``q·Y`` here;
    2. casting the fake key to the cache dtype;
    3. computing ``q·k`` in attention;
    4. multiplying ``q·k`` by the attention scale.

    For an n-term dot product, ``gamma_n = n*u/(1-n*u)`` bounds its relative error,
    where ``u = finfo.eps/2``. Multiplying this combined relative bound by
    ``sum(abs(q_i * Y_i))`` gives an absolute error bound even when terms cancel.
    Subtracting it from the measured margin gives the safe lower bound.
    """
    unit_roundoff = torch.finfo(X.dtype).eps / 2
    dot_product_error = X.shape[-1] * unit_roundoff / (1 - X.shape[-1] * unit_roundoff)
    key_cast_error = torch.finfo(output_dtype).eps / 2

    # There are two dot products (steps 1 and 3), one key cast, and one scalar multiply.
    combined_relative_error = (1 + dot_product_error) ** 2 * (1 + key_cast_error) * (1 + unit_roundoff) - 1

    margins = torch.bmm(X, Y.unsqueeze(-1)).squeeze(-1)
    sum_absolute_products = (X * Y.unsqueeze(1)).abs().sum(dim=-1)
    rounding_error_bound = combined_relative_error * sum_absolute_products
    return margins - rounding_error_bound


def _build_finite_fake_keys(X, Y, output_dtype, attention_scaling):
    """
    Turn a separating direction Y into finite fake keys.

    In a nutshell:
    1. Normalize Y without changing its direction.
    2. Find the query with the smallest positive dot product with Y.
    3. Negate and uniformly scale Y so even that worst-case query gets zero attention weight.
    4. Cast the result back to the cache dtype.

    Steps 1-3 run in float32 for fp16/bfloat16 inputs. Float16 can only represent magnitudes up
    to 65,504, but the fake keys may be larger. They must therefore be rescaled in float32 before
    being cast back; otherwise they become infinite and can produce NaN attention logits.

    The zero-attention threshold is derived from the smallest positive representable value,
    ``finfo.tiny * finfo.eps``. The margins include the standard dot-product rounding bound
    ``gamma_n = n * u / (1 - n * u)``, where ``u = finfo.eps / 2``.
    """
    # Compute the fake key in float32 for low-precision caches, then cast only the final result.
    scaling_dtype = torch.float32 if output_dtype in (torch.float16, torch.bfloat16) else output_dtype
    X = X.to(scaling_dtype)
    Y = Y.to(scaling_dtype)

    # Uniform normalization preserves the separating direction.
    Y = Y / Y.abs().amax(dim=-1, keepdim=True)

    safe_margins = _compute_safe_margins(X, Y, output_dtype)
    if (safe_margins <= 0).any():
        raise ValueError("The hyperplane is not separable with sufficient numerical precision")

    # Use one scale per batch row, based on its worst-case query. This preserves Y's direction while
    # making exp(attention_scaling * q·k) round to zero for every query in that row.
    softmax_finfo = torch.finfo(scaling_dtype)
    underflow_boundary = -math.log(softmax_finfo.tiny) - math.log(softmax_finfo.eps) + math.log(2)
    magnitude = underflow_boundary / (attention_scaling * safe_margins.amin(dim=-1, keepdim=True))
    if (magnitude > torch.finfo(output_dtype).max).any():
        raise ValueError("The fake keys required to mask attention are not representable")
    return (-magnitude * Y).to(output_dtype)
