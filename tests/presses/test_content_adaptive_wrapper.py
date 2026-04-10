# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress import KnormPress, SnapKVPress
from kvpress.presses.content_adaptive_wrapper import ContentAdaptiveWrapper
from tests.fixtures import unit_test_model  # noqa: F401


class TestContentAdaptiveWrapper:
    def test_requires_scorer_press(self):
        with pytest.raises(AssertionError, match="requires a ScorerPress"):
            ContentAdaptiveWrapper(press="not_a_press")

    def test_compression_ratio_delegation(self):
        base = KnormPress(compression_ratio=0.3)
        wrapper = ContentAdaptiveWrapper(press=base)
        assert wrapper.compression_ratio == 0.3
        wrapper.compression_ratio = 0.6
        assert base.compression_ratio == 0.6

    def test_detect_sets_content_type(self):
        base = KnormPress(compression_ratio=0.5)
        wrapper = ContentAdaptiveWrapper(press=base)
        result = wrapper.detect("def foo():\n    return 1\n" * 10)
        assert result == "code"
        assert wrapper.content_type == "code"

    @pytest.mark.parametrize(
        "base_cls",
        [KnormPress, SnapKVPress],
    )
    def test_wrapper_runs_with_model(self, unit_test_model, base_cls):  # noqa: F811
        if base_cls == SnapKVPress:
            base = base_cls(compression_ratio=0.5, window_size=2)
        else:
            base = base_cls(compression_ratio=0.5)
        wrapper = ContentAdaptiveWrapper(press=base, content_type="code")
        with wrapper(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache)
            assert cache.get_seq_length() == 32

    def test_wrapper_differs_from_base(self, unit_test_model):  # noqa: F811
        """Wrapper should produce different results than the base press alone."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        # Base press alone
        base = KnormPress(compression_ratio=0.5)
        with base(unit_test_model):
            cache_base = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_base)

        # Wrapped with code content type (strong boosts)
        base2 = KnormPress(compression_ratio=0.5)
        wrapper = ContentAdaptiveWrapper(press=base2, content_type="code")
        with wrapper(unit_test_model):
            cache_wrapped = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_wrapped)

        # At least one layer should differ
        any_different = False
        for i in range(len(cache_base.layers)):
            if not torch.equal(cache_base.layers[i].keys, cache_wrapped.layers[i].keys):
                any_different = True
                break
        assert any_different, "Wrapper produced identical output to base press"

    def test_zero_compression_is_identity(self, unit_test_model):  # noqa: F811
        base = KnormPress(compression_ratio=0.0)
        wrapper = ContentAdaptiveWrapper(press=base, content_type="code")
        with wrapper(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache)
            assert cache.get_seq_length() == 64

    def test_has_compression_ratio(self):
        base = KnormPress(compression_ratio=0.3)
        wrapper = ContentAdaptiveWrapper(press=base)
        assert hasattr(wrapper, "compression_ratio")
        assert wrapper.compression_ratio == 0.3
