# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import inspect
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, rotate_half

from kvpress import KeyRerotationPress, ScorerPress
from tests.fixtures import unit_test_model  # noqa: F401


@pytest.mark.parametrize("precision", ["full", "half"])
def test_rerotate_keys_is_matches_reference_implementation(unit_test_model: LlamaForCausalLM, precision):  # noqa: F811
    """
    Compare KeyRerotationPress' rerotation of keys with the reference implementation.
    In the reference implementation, we are computing
      1. keys = W_k * hidden_states
      2. keys_pruned = prune(keys)
      3. keys = RoPE(keys_pruned)
    """
    if precision == "half" and torch.cuda.is_available():
        unit_test_model = unit_test_model.cuda().half()
    elif precision == "half" and not torch.cuda.is_available():
        pytest.skip("Half precision test is skipped because CUDA is not available.")

    original_press = RandomPressStoreIndices(compression_ratio=0.5)
    key_rerotation_press = KeyRerotationPress(press=original_press)

    with key_rerotation_press(unit_test_model):
        module = unit_test_model.model.layers[0].self_attn
        hidden_states = torch.randn(
            8, 64, module.config.hidden_size, device=unit_test_model.device, dtype=unit_test_model.dtype
        )

        keys = get_keys_with_rope(module, hidden_states)

        values = torch.randn_like(keys)
        # Press result
        keys_compressed, _ = key_rerotation_press.compress(
            module,
            hidden_states,
            keys,
            values,
            attentions=None,
            kwargs={"position_embeddings": get_rope_embeddings(module, keys)},
        )

        indices = original_press.indices
        keys_compressed_ref = compute_rerotated_keys_comparison_implementation(module, hidden_states, indices)

    assert torch.allclose(keys_compressed, keys_compressed_ref, atol=1e-6 if precision == "full" else 1e-3)


def get_keys_with_rope(module, hidden_states):
    # Compute keys with RoPE
    keys = get_keys_without_pos_embedding(module, hidden_states)
    cos, sin = get_rope_embeddings(module, keys)
    keys = (keys * cos.unsqueeze(1)) + (rotate_half(keys) * sin.unsqueeze(1))
    return keys


@dataclass
class RandomPressStoreIndices(ScorerPress):
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
      1. keys = W_k * hidden_states
      2. keys_pruned = prune(keys)
      3. keys = RoPE(keys_pruned)
    """
    # 1.
    keys = get_keys_without_pos_embedding(module, hidden_states)
    # 2.
    keys = keys.gather(2, indices).contiguous()
    # 3.
    cos, sin = get_rope_embeddings(module, keys)
    keys = (keys * cos.unsqueeze(1)) + (rotate_half(keys) * sin.unsqueeze(1))
    return keys


def get_keys_without_pos_embedding(module, hidden_states):
    key_states = module.k_proj(hidden_states)
    key_states = key_states.view(
        key_states.shape[0], key_states.shape[1], module.config.num_key_value_heads, module.head_dim
    ).transpose(1, 2)
    return key_states


def get_rope_embeddings(module, x):
    length = x.shape[2]
    # rotary_emb function only needs .device and .dtype, so we can plug in any tensor regardless of shape
    if "position_ids" in inspect.signature(module.rotary_emb.forward).parameters:
        position_ids = torch.arange(length).unsqueeze(0).to(x.device)
        cos, sin = module.rotary_emb(x, position_ids)
    else:
        cos, sin = module.rotary_emb(x, length)
    return cos, sin
