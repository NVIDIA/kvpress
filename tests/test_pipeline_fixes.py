# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoTokenizer

from kvpress.pipeline import KVPressTextGenerationPipeline
from tests.fixtures import unit_test_model  # noqa: F401


def test_preprocess_without_chat_template_and_bos_token(unit_test_model):  # noqa: F811
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer.chat_template = None
    assert tokenizer.bos_token is None

    pipe = KVPressTextGenerationPipeline(model=unit_test_model, tokenizer=tokenizer, device=unit_test_model.device)
    inputs = pipe.preprocess("A short context.", ["A question?"], answer_prefix="", max_context_length=100)

    expected_ids = tokenizer.encode("A short context.", return_tensors="pt", add_special_tokens=False)
    assert torch.equal(inputs["context_ids"], expected_ids)
