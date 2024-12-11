# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, LlamaRotaryEmbedding, rotate_half

from kvpress import KeyRerotationPress, ScorerPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_rerotate_keys_is_matches_reference_implementation(unit_test_model: LlamaForCausalLM):  # noqa: F811
    """
    Compare KeyRerotationPress' rerotation of keys with the reference implementation.
    In KeyRerotationPress, we are using trigonometric functions to rerotate the keys.
    In the reference implementation, we are using the
    """
    original_press = RandomPressWithSeed(compression_ratio=0.5)
    key_rerotation_press = KeyRerotationPress(press=original_press)

    module = unit_test_model.model.layers[0].self_attn
    hidden_states = torch.randn(8, 64, module.config.hidden_size)

    keys = get_keys_with_rope(module, hidden_states)

    values = torch.randn_like(keys)
    keys_compressed, _ = key_rerotation_press.compress(
        module, hidden_states, keys, values, attentions=None, kwargs=dict()
    )

    indices = original_press.indices
    keys_compressed_ref = compute_rerotated_keys_comparison_implementation(module, hidden_states, indices)

    assert torch.allclose(keys_compressed, keys_compressed_ref, atol=1e-6)


def get_keys_with_rope(module, hidden_states):
    # Compute keys with RoPE
    keys = get_keys_without_pos_embedding(module, hidden_states)
    cos, sin = get_position_embeddings(keys, module.rotary_emb)
    keys = (keys * cos.unsqueeze(1)) + (rotate_half(keys) * sin.unsqueeze(1))
    return keys


@dataclass
class RandomPressWithSeed(ScorerPress):
    compression_ratio: float = 0.0
    seed: int = 0

    def __post_init__(self):
        self.indices = None
        super().__post_init__()

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        torch.manual_seed(self.seed)
        scores = torch.rand(*keys.shape[:-1]).to(keys.device, keys.dtype)
        # Get indices of KV pairs with the lowest scores
        q_len = hidden_states.shape[1]
        n_kept = int(q_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        self.indices = indices

        return scores


def compute_rerotated_keys_comparison_implementation(module: LlamaAttention, hidden_states, indices):
    """
    Computes the rerotated keys for the given indices.
    This is a reference implementation for the rerotation of keys.
    """
    keys = get_keys_without_pos_embedding(module, hidden_states)

    keys = keys.gather(2, indices).contiguous()
    # apply position embeddings on the pruned keys
    cos, sin = get_position_embeddings(keys, module.rotary_emb)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    keys = (keys * cos) + (rotate_half(keys) * sin)
    return keys


def get_keys_without_pos_embedding(module, hidden_states):
    key_states = module.k_proj(hidden_states)
    key_states = key_states.view(
        key_states.shape[0], key_states.shape[1], module.num_key_value_heads, module.head_dim
    ).transpose(1, 2)
    return key_states


def get_position_embeddings(keys, rotary_emb: LlamaRotaryEmbedding):
    length = keys.shape[2]
    position_ids = torch.arange(length).unsqueeze(0).to(keys.device)
    cos, sin = rotary_emb(keys, position_ids)
    return cos, sin
