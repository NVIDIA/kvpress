import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def search_hyperplane(X, max_iter: int = 1000, epsilon: float = 1e-5):
    """
    Search for an hyperplane Y such that for every Xi, <Xi, Y> <= epsilon (simple perceptron)
    Returns - Y / espilon ** 2 to ensure exp(<X, Y>) = 0
    """
    Y = X.mean(1)
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= epsilon
        if not mask.any():
            return -Y / epsilon**2
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError("Could not find fake keys such that for every query q, exp(<q, k>) = 0")


def attention_patch(func):
    """
    Decorator to udpate the keys before the attention computation at the indices provided in module.indices
    The keys are updated to a fake key k such that for the input queries q, exp(<q, k>) = 0
    This is used to fake head-wise compression. A more optimal solution would be to create a new kernel.
    """

    def wrapper(module, query, key, value, attention_mask, dropout, scaling=None, is_causal=None, **kwargs):
        if query.shape[2] == key.shape[2]:
            # Prefilling
            module.indices = None
        elif module.indices is not None:
            # Decoding: build fake keys k s.t. exp(<q, k>) = 0
            bsz, num_heads, seq_len, head_dim = query.shape
            num_key_value_heads = key.shape[1]
            num_groups = num_heads // num_key_value_heads

            # Build a fake key k per key group such that for every query q, exp(<q, k>) = 0
            q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
            q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
            k = search_hyperplane(q)
            k = k.view(bsz, num_key_value_heads, head_dim)

            # At indices, update the keys to the fake keys and the values to 0
            key[*module.indices] = k[*module.indices[:2]]

        return func(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)

    return wrapper


def patch_attention_functions():
    """
    Add the attention_patch decorator to functions in ALL_ATTENTION_FUNCTIONS
    """

    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        ALL_ATTENTION_FUNCTIONS[name] = attention_patch(func)
