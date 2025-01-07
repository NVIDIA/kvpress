import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

LARGE_NEGATIVE_FLOAT = -float(1e5)


def attention_patch(func):
    """
    Decorator to udpate the keys and values before the attention computation at the indices provided in module.indices
    The keys are updated to a fake key k such that for the input queries q, exp(<q, k>) = 0. The values are set to 0.
    This is used to fake head-wise compression. A more optimal solution would be to create a new kernel.
    """

    def wrapper(module, query, key, value, attention_mask, dropout, scaling=None, is_causal=None, **kwargs):
        if query.shape[2] == key.shape[2]:
            # Prefilling phase
            module.indices = None
        elif module.indices is not None:
            # Decoding phase
            bsz, num_heads, seq_len, head_dim = query.shape
            num_key_value_heads = key.shape[1]
            num_groups = num_heads // num_key_value_heads

            # Build a fake key k per key group such that for every query q, exp(<q, k>) = 0
            # To do so, use the least square method to find k such that q @ k ~ LARGE_NEGATIVE_FLOAT
            q = query.view(bsz, num_groups, num_key_value_heads, seq_len, head_dim)
            q = q.transpose(1, 2).reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
            targets = LARGE_NEGATIVE_FLOAT * torch.ones(q.shape[:2]).to(q.device)
            k = torch.linalg.lstsq(q.float(), targets)[0].to(q.dtype)
            assert torch.exp(torch.einsum("hnd,hd->hn", q, k).max()) == 0, "Could not find fake keys"
            k = k.view(bsz, num_key_value_heads, head_dim)

            # At indices, update the keys to the fake keys and the values to 0
            key[*module.indices] = k[*module.indices[:2]]
            value[*module.indices] = 0  # TODO: do this only once in the forward_hook ?

        return func(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)

    return wrapper


def patch_attention_functions():
    """
    Add the update_keys_before_attention decorator to all attention functions in ALL_ATTENTION_FUNCTIONS
    """

    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        ALL_ATTENTION_FUNCTIONS[name] = attention_patch(func)
