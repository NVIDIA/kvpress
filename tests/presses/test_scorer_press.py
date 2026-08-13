# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest
import torch

from kvpress.presses.adakv_press import AdaKVPress
from kvpress.presses.block_press import BlockPress
from kvpress.presses.criticalkv_press import CriticalAdaKVPress
from kvpress.presses.finch_press import FinchPress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.merging_press import MergingPress

# Short contexts where ``int(k_len * (1 - compression_ratio))`` floors to 0.
# Without a floor guard the cache is emptied to (bsz, heads, 0, head_dim) and
# the next decode silently attends zero keys (NaN / divergent generation). The
# ``max(1, int(...))`` guard keeps at least one token at every n_kept site.
PRESS_NAMES = [
    "KnormPress",
    "BlockPress",
    "MergingPress",
    "KeyRerotationPress",
    "FinchPress",
    "AdaKVPress",
    "CriticalAdaKVPress",
]


def _make_press(name, ratio):
    child = KnormPress(compression_ratio=ratio)
    if name == "KnormPress":
        return child
    if name == "BlockPress":
        return BlockPress(press=child)
    if name == "MergingPress":
        return MergingPress(press=child)
    if name == "KeyRerotationPress":
        return KeyRerotationPress(press=child)
    if name == "FinchPress":
        press = FinchPress(compression_ratio=ratio, rerotate_keys=False)
        press.window_size = 1
        return press
    if name == "AdaKVPress":
        return AdaKVPress(press=child)
    if name == "CriticalAdaKVPress":
        return CriticalAdaKVPress(press=child)
    raise ValueError(name)


def _make_module(name, head_dim, num_key_value_heads):
    """Minimal fake attention module exposing only what each press reads."""
    module = types.SimpleNamespace(head_dim=head_dim)
    if name == "KeyRerotationPress":
        # rerotate_keys reads module.rotary_emb.inv_freq
        module.rotary_emb = types.SimpleNamespace(inv_freq=torch.randn(head_dim // 2))
    if name == "FinchPress":
        module.config = types.SimpleNamespace(num_attention_heads=num_key_value_heads)
    if name in ("AdaKVPress", "CriticalAdaKVPress"):
        num_attention_heads = num_key_value_heads  # num_key_value_groups == 1
        hidden_size = num_attention_heads * head_dim
        module.config = types.SimpleNamespace(
            _attn_implementation="sdpa",
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
        )
        module.num_key_value_groups = 1
        module.o_proj = types.SimpleNamespace(
            weight=torch.randn(hidden_size, num_attention_heads * head_dim)
        )
    return module


@pytest.mark.parametrize("k_len, ratio", [(1, 0.5), (2, 0.6)])
@pytest.mark.parametrize("press_name", PRESS_NAMES)
def test_compress_never_empties_cache_on_short_context(press_name, k_len, ratio):
    """Short contexts must never collapse the cache to zero tokens.

    Reproduces the n_kept floor-to-zero bug: ``int(k_len * (1 - ratio))`` is 0
    for the parametrized cases, so the cache is emptied (shape[2] == 0) or every
    token is flagged for pruning. The ``max(1, ...)`` guard keeps at least one
    token, so the compressed cache stays non-empty and finite.
    """
    if press_name == "FinchPress" and k_len < 2:
        # FinchPress windowing needs a window strictly shorter than the sequence.
        pytest.skip("FinchPress requires k_len > window_size")

    bsz, num_key_value_heads, head_dim = 1, 2, 8
    hidden_dim = num_key_value_heads * head_dim
    keys = torch.randn(bsz, num_key_value_heads, k_len, head_dim)
    values = torch.randn(bsz, num_key_value_heads, k_len, head_dim)
    hidden_states = torch.randn(bsz, k_len, hidden_dim)

    module = _make_module(press_name, head_dim, num_key_value_heads)
    press = _make_press(press_name, ratio)

    if press_name == "FinchPress":
        num_heads = num_key_value_heads  # num_key_value_groups == 1
        attentions = torch.randn(bsz, num_heads, press.window_size, k_len)
    else:
        attentions = None
    kwargs = {}

    out_keys, _ = press.compress(module, hidden_states, keys, values, attentions, kwargs)

    # The cache must never be emptied to zero tokens.
    assert out_keys.shape[2] >= 1
    assert not torch.isnan(out_keys).any()

    # AdaKVPress / CriticalAdaKVPress return keys unchanged but flag pruned
    # tokens via module.masked_key_indices; with the bug every token is pruned.
    masked = getattr(module, "masked_key_indices", None)
    if masked is not None:
        assert len(masked[2]) < num_key_value_heads * k_len
