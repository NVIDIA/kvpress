from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

LARGE_NEGATIVE_FLOAT = -1e5


def search_hyperplane(X, max_iter=1000):
    """
    Search for an hyperplane Y such that for every Xi, <Xi, Y> <= 1 (simple perceptron)
    Returns LARGE_NEGATIVE_FLOAT * Y to ensure exp(<X, Y>) = 0
    """
    Y = X.mean(1)
    for _ in range(max_iter):
        mask = (X * Y.unsqueeze(1)).sum(dim=2, keepdim=True) <= 1
        if not mask.any():
            return LARGE_NEGATIVE_FLOAT * Y
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
            # Prefilling phase
            module.indices = None
        elif module.indices is not None:
            # Decoding phase
            bsz, num_heads, seq_len, head_dim = query.shape

            # Build a fake key k per key group such that for every query q, exp(<q, k>) = 0
            q = query.reshape(bsz * num_heads, seq_len, head_dim)
            k = search_hyperplane(q)
            k = k.view(bsz, num_heads, head_dim)

            # At indices, update the keys to the fake keys
            key[*module.indices] = k[*module.indices[:2]]

        return func(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)

    return wrapper


def patch_attention_functions():
    """
    Add the update_keys_before_attention decorator to all attention functions in ALL_ATTENTION_FUNCTIONS
    """

    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        ALL_ATTENTION_FUNCTIONS[name] = attention_patch(func)
