# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from kvpress import CompactorPress, LeverageScorePress, NonCausalAttnPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_compactor_press(unit_test_model):  # noqa: F811
    for press in [
        CompactorPress(0.5, sink_size_start=0, sink_size_end=0),
        CompactorPress(0.2, sink_size_start=8, sink_size_end=4),
    ]:
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)


def test_leverage_press(unit_test_model):  # noqa: F811
    for press in [
        LeverageScorePress(0.5, sketch_dimension=48),
        LeverageScorePress(0.5, sketch_dimension=64),
    ]:
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)


def test_non_causal_attn_press(unit_test_model):  # noqa: F811
    for press in [
        NonCausalAttnPress(0.5, chunk_size=128),
        NonCausalAttnPress(0.5, chunk_size=256),
    ]:
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)


def test_non_causal_chunked_attn_does_not_leak_mass_to_padding_keys():
    # S=2 with chunk_size=4 puts the whole (single) chunk in the ragged case: 2 real
    # query/key positions plus 2 padded ones. Every one of the 4 query rows in that
    # chunk (real or padded) puts its full softmax unit of mass on the 2 real key
    # columns when padding is masked out correctly, so the returned per-key scores
    # (summed over all 4 queries) must add up to ~chunk_size regardless of the actual
    # q/k values. A masking value too small to dominate the logits lets softmax leak
    # part of that mass onto the phantom zero-vector padding keys instead, which are
    # then dropped, so the real keys' scores come back under-counted.
    torch.manual_seed(0)
    B, H, S, d, chunk_size = 1, 1, 2, 4, 4
    q = torch.randn(B, H, S, d)
    k = torch.randn(B, H, S, d)

    scores = NonCausalAttnPress.non_causal_chunked_attn(q, k, chunk_size)
    assert scores.sum().item() == pytest.approx(chunk_size, abs=1e-3)
