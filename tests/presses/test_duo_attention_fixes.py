# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kvpress.presses.duo_attention_press import PATTERNS_DICT


def test_duo_attention_patterns_match_hub_checkpoints():
    assert all(name.count("/") == 1 for name in PATTERNS_DICT)
    assert "gradientai/Llama-3-8B-Instruct-Gradient-1048k" in PATTERNS_DICT
    assert PATTERNS_DICT["meta-llama/Llama-3.1-8B-Instruct"] == PATTERNS_DICT["meta-llama/Meta-Llama-3.1-8B-Instruct"]
