## Registering new attention modules

The `modeling_{name}.py` files have been created running the `utils.rewrite_modeling_scripts` function. This function simply adds the `query_states` in the `cache_kwargs`.

The `utils.update_attn_implementations` function can then bue used to register the attention classes in the `{NAME}_ATTENTION_CLASSES` dictionary.