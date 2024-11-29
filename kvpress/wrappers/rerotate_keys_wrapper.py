# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import inspect

import torch
import torch.nn as nn
from transformers import QuantizedCache
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.base_press import BasePress


def apply_key_rerotation(press: BasePress) -> BasePress:
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
    assert (
            press.wrappers_applied == []
    ), f"apply_key_rerotation must be the first wrapper applied. Already applied: {press.wrappers_applied}"
    press.wrappers_applied.append("apply_key_rerotation")

    press.forward_hook = forward_rerotate_hook.__get__(press, BasePress)  # type: ignore[method-assign]
    return press


def forward_rerotate_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
    """
    Forward hook that rerotates keys after pruning. See `BasePress.forward_hook` for more details.

    Parameters
    ----------
    module :
        Transformer attention layer.
    input :
        Input to the hook. This is the input to the forward pass of the layer.
    kwargs :
        Keyword arguments, as given to the forward pass of the layer.
    output :
        Output of the hook. This is the original output of the forward pass of the layer.

    Returns
    -------
        Modified output of the forward pass of the layer.

    """
    # See e.g. LlamaDecoderLayer.forward for the output structure
    if len(output) == 3:
        _, attentions, cache = output
    else:
        attentions, cache = None, output[-1]

    hidden_states = kwargs["hidden_states"]
    q_len = hidden_states.shape[1]

    # Don't compress if the compression ratio is 0 or this is not pre-filling
    if (self.compression_ratio == 0) or (cache.seen_tokens > q_len):
        return output

    if isinstance(cache, QuantizedCache):
        keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])
        values = cache._dequantize(cache._quantized_value_cache[module.layer_idx])
    else:
        keys = cache.key_cache[module.layer_idx]
        values = cache.value_cache[module.layer_idx]

    with torch.no_grad():
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

    # Prune KV pairs with the lowest scores
    n_kept = int(q_len * (1 - self.compression_ratio))
    indices = scores.topk(n_kept, dim=-1).indices
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

    # Apply RoPE to the pruned keys
    keys = get_keys_without_pos_embedding(module, hidden_states)
    keys = keys.gather(2, indices).contiguous()
    # apply position embeddings on the pruned keys
    cos, sin = get_position_embeddings(module, keys)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    keys = (keys * cos) + (rotate_half(keys) * sin)

    values = values.gather(2, indices).contiguous()
    if isinstance(cache, QuantizedCache):
        cache._quantized_key_cache[module.layer_idx] = cache._quantize(keys, axis=cache.axis_key)
        cache._quantized_value_cache[module.layer_idx] = cache._quantize(values, axis=cache.axis_value)
    else:
        cache.key_cache[module.layer_idx] = keys
        cache.value_cache[module.layer_idx] = values

    return output


def get_keys_without_pos_embedding(module, hidden_states):
    if hasattr(module, "k_proj"):
        key_states = module.k_proj(hidden_states)

    elif hasattr(module, "qkv_proj"):
        # see modeling_phi3.py
        qkv = module.qkv_proj(hidden_states[:, -module.window_size:])
        query_pos = module.num_heads * module.head_dim
        key_states = qkv[..., query_pos: query_pos + module.num_key_value_heads * module.head_dim]
    else:
        raise NotImplementedError(f"ExpectedAttentionPress not yet implemented for {module.__class_}.")
    key_states = key_states.view(key_states.shape[0], key_states.shape[1], module.num_heads, module.head_dim).transpose(
        1, 2
    )
    return key_states


def get_position_embeddings(module, x):
    length = x.shape[2]
    # rotary_emb function only needs .device and .dtype, so we can plug in any tensor regardless of shape
    if "position_ids" in inspect.signature(module.rotary_emb.forward).parameters:
        position_ids = torch.arange(length).unsqueeze(0).to(x.device)
        cos, sin = module.rotary_emb(x, position_ids)
    else:
        cos, sin = module.rotary_emb(x, length)
    return cos, sin
