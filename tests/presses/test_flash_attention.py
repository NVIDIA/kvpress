# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoTokenizer
from transformers.utils import is_flash_attn_2_available

from kvpress import KnormPress
from tests.fixtures import kv_press_qwen3_flash_attn_pipeline  # noqa: F401


class TestFlashAttention:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
    @pytest.mark.skipif(not is_flash_attn_2_available(), reason="flash_attn is not installed")
    def test_fa_works(self, kv_press_qwen3_flash_attn_pipeline):  # noqa: F811
        # test if fa2 runs, see https://github.com/huggingface/transformers/releases/tag/v4.55.2
        # and https://github.com/NVIDIA/kvpress/pull/115
        model = kv_press_qwen3_flash_attn_pipeline.model
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
        inputs = tok("Hello, how are you? bla bla how are you? this is some text lala ddd", return_tensors="pt").to(
            model.device
        )

        with KnormPress(0.8)(model):
            model.generate(**inputs, max_new_tokens=10, do_sample=False)
