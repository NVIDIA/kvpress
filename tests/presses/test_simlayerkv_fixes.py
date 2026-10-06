# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress import SimLayerKVPress
from tests.fixtures import unit_test_model  # noqa: F401


@pytest.mark.parametrize("n_recent, n_last", [(1, 1), (1, 2), (4, 1), (6, 2)])
@torch.no_grad()
def test_lazy_layers_keep_initial_and_recent_tokens(unit_test_model, n_recent, n_last):  # noqa: F811
    input_ids = torch.randint(0, 1000, (1, 32), device=unit_test_model.device)
    reference_cache = DynamicCache()
    unit_test_model(input_ids, past_key_values=reference_cache)

    # lazy_threshold=0 makes every layer lazy
    press = SimLayerKVPress(lazy_threshold=0.0, n_initial=2, n_recent=n_recent, n_last=n_last)
    cache = DynamicCache()
    with press(unit_test_model):
        unit_test_model(input_ids, past_key_values=cache)

    n_recent_kept = max(0, n_recent - n_last)
    kept_positions = list(range(2)) + list(range(32 - n_recent_kept, 32))
    for layer, reference_layer in zip(cache.layers, reference_cache.layers):
        torch.testing.assert_close(layer.keys, reference_layer.keys[:, :, kept_positions])
    assert press.compression_ratio == pytest.approx(1 - len(kept_positions) / 32)
