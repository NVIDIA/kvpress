# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
from dataclasses import dataclass
from typing import Callable

import pytest
import torch

from kvpress.presses.adakv_press import AdaKVPress
from kvpress.presses.block_press import BlockPress
from kvpress.presses.criticalkv_press import CriticalAdaKVPress
from kvpress.presses.finch_press import FinchPress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.merging_press import MergingPress


@dataclass
class ShortContextCase:
    """A press plus the minimal fake module/attentions it needs.

    Each case owns its own construction so the parametrization extends by
    appending a case, not by editing shared if/elif string dispatch.
    """

    id: str
    make_press: Callable[[float], object]
    make_module: Callable[[int, int], object]
    make_attentions: Callable[[int, int, int], object]


def _simple_module(head_dim, num_key_value_heads):
    return types.SimpleNamespace(head_dim=head_dim)


def _rerotation_module(head_dim, num_key_value_heads):
    module = _simple_module(head_dim, num_key_value_heads)
    module.rotary_emb = types.SimpleNamespace(inv_freq=torch.randn(head_dim // 2))
    return module


def _finch_module(head_dim, num_key_value_heads):
    module = _simple_module(head_dim, num_key_value_heads)
    module.config = types.SimpleNamespace(num_attention_heads=num_key_value_heads)
    return module


def _adakv_module(head_dim, num_key_value_heads):
    num_attention_heads = num_key_value_heads  # num_key_value_groups == 1
    hidden_size = num_attention_heads * head_dim
    module = _simple_module(head_dim, num_key_value_heads)
    module.config = types.SimpleNamespace(
        _attn_implementation="sdpa",
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
    )
    module.num_key_value_groups = 1
    module.o_proj = types.SimpleNamespace(weight=torch.randn(hidden_size, num_attention_heads * head_dim))
    return module


def _no_attentions(bsz, num_key_value_heads, k_len):
    return None


def _finch_attentions(bsz, num_key_value_heads, k_len):
    num_heads = num_key_value_heads  # num_key_value_groups == 1
    return torch.randn(bsz, num_heads, 1, k_len)


def _finch_press(ratio):
    press = FinchPress(compression_ratio=ratio, rerotate_keys=False)
    press.window_size = 1
    return press


# Short contexts where ``int(k_len * (1 - compression_ratio))`` floors to 0.
# Without a floor guard the cache is emptied to (bsz, heads, 0, head_dim) and
# the next decode silently attends zero keys. The ``max(1, int(...))`` guard
# keeps at least one token at every n_kept site.
SHORT_CONTEXT_CASES = [
    ShortContextCase("knorm", lambda r: KnormPress(compression_ratio=r), _simple_module, _no_attentions),
    ShortContextCase(
        "block", lambda r: BlockPress(press=KnormPress(compression_ratio=r)), _simple_module, _no_attentions
    ),
    ShortContextCase(
        "merging", lambda r: MergingPress(press=KnormPress(compression_ratio=r)), _simple_module, _no_attentions
    ),
    ShortContextCase(
        "key_rerotation",
        lambda r: KeyRerotationPress(press=KnormPress(compression_ratio=r)),
        _rerotation_module,
        _no_attentions,
    ),
    ShortContextCase("finch", _finch_press, _finch_module, _finch_attentions),
    ShortContextCase(
        "adakv", lambda r: AdaKVPress(press=KnormPress(compression_ratio=r)), _adakv_module, _no_attentions
    ),
    ShortContextCase(
        "critical_adakv",
        lambda r: CriticalAdaKVPress(press=KnormPress(compression_ratio=r)),
        _adakv_module,
        _no_attentions,
    ),
]


@pytest.mark.parametrize("k_len, ratio", [(1, 0.5), (2, 0.6)])
@pytest.mark.parametrize("case", SHORT_CONTEXT_CASES, ids=lambda c: c.id)
def test_compress_never_empties_cache_on_short_context(case, k_len, ratio):
    """Short contexts must never collapse the cache to zero tokens.

    ``int(k_len * (1 - ratio))`` is 0 for the parametrized cases, so without
    the guard the cache is emptied (shape[2] == 0) or every token is flagged
    for pruning. The ``max(1, ...)`` guard keeps at least one token, so the
    compressed cache stays non-empty.
    """
    if case.id == "finch" and k_len < 2:
        pytest.skip("FinchPress requires k_len > window_size")

    bsz, num_key_value_heads, head_dim = 1, 2, 8
    hidden_dim = num_key_value_heads * head_dim
    keys = torch.randn(bsz, num_key_value_heads, k_len, head_dim)
    values = torch.randn(bsz, num_key_value_heads, k_len, head_dim)
    hidden_states = torch.randn(bsz, k_len, hidden_dim)

    module = case.make_module(head_dim, num_key_value_heads)
    press = case.make_press(ratio)
    attentions = case.make_attentions(bsz, num_key_value_heads, k_len)

    out_keys, _ = press.compress(module, hidden_states, keys, values, attentions, {})

    # The cache must never be emptied to zero tokens.
    assert out_keys.shape[2] >= 1

    # AdaKVPress / CriticalAdaKVPress return keys unchanged but flag pruned
    # tokens via module.masked_key_indices; with the bug every token is pruned.
    masked = getattr(module, "masked_key_indices", None)
    if masked is not None:
        assert len(masked[2]) < num_key_value_heads * k_len


@pytest.mark.parametrize("press_cls", [AdaKVPress, CriticalAdaKVPress])
def test_adakv_safeguard_protects_every_head(press_cls):
    """The per-head ``alpha_safeguard`` must protect at least one token per head.

    With ``n_kept=1`` and the default ``alpha_safeguard=0.2``, ``n_safe`` floors
    to 0. Without the ``max(1, n_safe)`` clamp no token is protected per head,
    so a head whose keys all score lowest can be fully pruned. Build that
    adversarial case (one high-norm head, one zero-norm head) and assert no
    head ends with all ``k_len`` positions masked.
    """
    bsz, num_key_value_heads, k_len, head_dim = 1, 2, 2, 8
    ratio = 0.6  # n_kept = max(1, int(2 * 0.4)) = 1; n_safe = int(1 * 0.2) = 0 without clamp

    keys = torch.zeros(bsz, num_key_value_heads, k_len, head_dim)
    keys[:, 0, :, :] = 10.0  # head 0: high norm -> lowest score -> pruned without safeguard
    values = torch.randn(bsz, num_key_value_heads, k_len, head_dim)
    hidden_states = torch.randn(bsz, k_len, num_key_value_heads * head_dim)

    module = _adakv_module(head_dim, num_key_value_heads)
    press = press_cls(press=KnormPress(compression_ratio=ratio))
    press.compress(module, hidden_states, keys, values, None, {})

    _, head_indices, _ = module.masked_key_indices
    per_head = torch.zeros(num_key_value_heads, dtype=torch.int64)
    per_head.scatter_add_(0, head_indices, torch.ones_like(head_indices))
    assert (per_head < k_len).all(), f"a head was fully masked: {per_head.tolist()}"
