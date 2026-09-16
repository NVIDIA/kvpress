# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# Casting the fake key to the cache dtype perturbs it by at most one unit roundoff in norm, i.e. 2**-8 ≈ 0.4%
# for bfloat16, the coarsest supported dtype. A query whose cosine with the hyperplane normal exceeds that keeps
# a negative logit after the cast. Requiring 1% leaves headroom for further rounding, e.g. TF32 matmuls.
MIN_COSINE = 0.01

# exp(-1000) is exactly zero in every floating point dtype up to float64 (whose exp underflows below -745), with
# ample headroom for rounding of the logit itself (e.g. in a bfloat16 matmul) or a negative maximum logit.
TARGET_LOGIT = -1000.0

# When a float16 cache cannot hold the key needed for TARGET_LOGIT, any logit below log(2**-150) ≈ -104 is still
# exactly zero after exp in float32, the softmax precision used by attention implementations for float16 models.
UNDERFLOW_LOGIT = -104.0


def search_hyperplane(X, max_iter: int = 1000, attention_scaling: float = 1.0):
    """
    Given a tensor X of shape (bsz, seq_len, head_dim), return fake keys K (bsz, head_dim)
    such that exp(attention_scaling * <X[:, i], K>) underflows to zero for every i.

    Starting from the mean query direction Y, repeatedly add the queries whose cosine with Y is
    below MIN_COSINE (a perceptron with margin). Once every query is safely on the positive side
    of the hyperplane, negate and scale Y so that every scaled attention logit is at most TARGET_LOGIT,
    or as negative as the cache dtype allows while still underflowing (see UNDERFLOW_LOGIT).

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
        If no valid hyperplane is found or the required fake key is not representable in X.dtype.
    """
    if attention_scaling <= 0:
        raise ValueError("attention_scaling must be positive")

    output_dtype = X.dtype
    # Search in float32: float16 overflows as Y grows, and neither float16 nor bfloat16 can resolve MIN_COSINE.
    if output_dtype in (torch.float16, torch.bfloat16):
        X = X.float()

    query_norms = X.norm(dim=-1)  # (bsz, seq_len)
    Y = X.mean(1)  # this initialization is enough for most cases
    for _ in range(max_iter):
        margins = torch.bmm(X, Y.unsqueeze(-1)).squeeze(-1)  # (bsz, seq_len)
        violating = margins <= MIN_COSINE * query_norms * Y.norm(dim=-1, keepdim=True)
        if not violating.any():
            return _build_fake_keys(X, Y, output_dtype, attention_scaling)
        Y += (X * violating.unsqueeze(-1)).sum(1) / violating.sum(1, keepdim=True).clamp(min=1)
    raise ValueError("Could not find a hyperplane that nullifies every query")


def _build_fake_keys(X, Y, output_dtype, attention_scaling):
    """
    Negate and scale the separating direction Y so that attention_scaling * <X[:, i], K> <= TARGET_LOGIT for
    every query, while keeping K representable in output_dtype (float16 overflows above 65,504).
    X and Y must be float32 or float64: only the final result is cast.
    """
    # Scale Y so that its largest component is 1: the fake key's largest component is then `magnitude`.
    Y = Y / Y.abs().amax(dim=-1, keepdim=True)
    min_margin = torch.bmm(X, Y.unsqueeze(-1)).amin(dim=1)  # (bsz, 1), positive since the search converged
    magnitude = -TARGET_LOGIT / (attention_scaling * min_margin)
    magnitude = magnitude.clamp(max=torch.finfo(output_dtype).max)
    if (-attention_scaling * min_margin * magnitude > UNDERFLOW_LOGIT).any():
        raise ValueError(f"The fake keys required to mask attention are not representable in {output_dtype}")
    return (-magnitude * Y).to(output_dtype)


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
