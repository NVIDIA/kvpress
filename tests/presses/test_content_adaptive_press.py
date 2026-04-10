# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress.presses.content_adaptive_press import (
    CONTENT_PARAMS,
    ContentAdaptivePress,
    classify_content,
)
from tests.fixtures import unit_test_model  # noqa: F401


# ---------------------------------------------------------------------------
# classify_content tests
# ---------------------------------------------------------------------------


class TestClassifyContent:
    def test_code_detection(self):
        text = (
            'def hello(): return "world"\n'
            'class Foo: pass\n'
            'import os\n'
            'def __init__(self): pass\n'
            'function bar() { return 1; }\n'
            'const x = 42; var y = 0; let z = 1;\n'
            'if (true) { for (int i = 0; i < 10; i++) {} }\n'
            'try: except: pass\n'
        )
        assert classify_content(text) == "code"

    def test_math_detection(self):
        text = r"""
Consider the equation \sum_{i=1}^{n} x_i = 0.
By the \int theorem and \frac{d}{dx} we get \sqrt{2}.
\begin{equation} proof of the lemma follows.
"""
        assert classify_content(text) == "math"

    def test_prose_detection(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a normal English sentence with punctuation. "
            "Another sentence follows here, describing events in detail. "
            "The narrative continues with more prose content."
        )
        assert classify_content(text) == "prose"

    def test_structured_detection(self):
        # Many lines per sentence → structured
        text = "\n".join([f"row_{i}: value_{i}" for i in range(100)])
        assert classify_content(text) == "structured"

    def test_empty_string(self):
        assert classify_content("") == "prose"

    def test_short_string(self):
        assert classify_content("hello") == "prose"


# ---------------------------------------------------------------------------
# ContentAdaptivePress tests
# ---------------------------------------------------------------------------


class TestContentAdaptivePress:
    def test_detect_sets_content_type(self):
        press = ContentAdaptivePress(compression_ratio=0.5)
        result = press.detect("def foo():\n    return 1\n" * 10)
        assert result == "code"
        assert press.content_type == "code"

    def test_detect_returns_type(self):
        press = ContentAdaptivePress(compression_ratio=0.5)
        ct = press.detect("Some normal prose text. Another sentence here.")
        assert ct == press.content_type

    @pytest.mark.parametrize("content_type", list(CONTENT_PARAMS.keys()))
    def test_score_runs_with_model(self, unit_test_model, content_type):  # noqa: F811
        press = ContentAdaptivePress(compression_ratio=0.5, content_type=content_type)
        with press(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)
            cache = DynamicCache()
            output = unit_test_model(input_ids, past_key_values=cache)
            # Cache should be compressed to ~50%
            assert cache.get_seq_length() == 32

    def test_different_content_types_give_different_results(self, unit_test_model):  # noqa: F811
        """Different content types should produce different cache contents."""
        results = {}
        for ct in ["code", "math", "prose"]:
            press = ContentAdaptivePress(compression_ratio=0.5, content_type=ct)
            with press(unit_test_model):
                torch.manual_seed(42)
                input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)
                cache = DynamicCache()
                unit_test_model(input_ids, past_key_values=cache)
                results[ct] = cache.layers[0].keys.clone()  # first layer keys

        # At least one pair should differ (different boost profiles)
        any_different = False
        types = list(results.keys())
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                if not torch.equal(results[types[i]], results[types[j]]):
                    any_different = True
                    break
        assert any_different, "All content types produced identical cache — boosts had no effect"

    def test_compression_ratio_zero_is_identity(self, unit_test_model):  # noqa: F811
        press = ContentAdaptivePress(compression_ratio=0.0)
        with press(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache)
            assert cache.get_seq_length() == 64

    def test_has_compression_ratio(self):
        press = ContentAdaptivePress(compression_ratio=0.3)
        assert press.compression_ratio == 0.3
