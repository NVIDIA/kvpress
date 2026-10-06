# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoTokenizer

from kvpress import DecodingPress, DMSPress, KnormPress, PrefillDecodingPress, RandomPress
from kvpress.pipeline import KVPressTextGenerationPipeline
from tests.fixtures import kv_press_unit_test_pipeline, unit_test_model  # noqa: F401

CONTEXT = "This is a test article. It was written on 2022-01-01. " * 4
QUESTIONS = ["When was this article written?", "What is this?"]


def test_preprocess_without_chat_template_and_bos_token(unit_test_model):  # noqa: F811
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer.chat_template = None
    assert tokenizer.bos_token is None

    pipe = KVPressTextGenerationPipeline(model=unit_test_model, tokenizer=tokenizer, device=unit_test_model.device)
    inputs = pipe.preprocess("A short context.", ["A question?"], answer_prefix="", max_context_length=100)

    expected_ids = tokenizer.encode("A short context.", return_tensors="pt", add_special_tokens=False)
    assert torch.equal(inputs["context_ids"], expected_ids)


@pytest.mark.parametrize(
    "make_press",
    [
        lambda: DecodingPress(base_press=KnormPress(), compression_interval=2, target_size=16),
        lambda: PrefillDecodingPress(
            decoding_press=DecodingPress(base_press=KnormPress(), compression_interval=2, target_size=16)
        ),
        lambda: DMSPress(press=RandomPress(), threshold=0.5, sliding_window_size=0, decoding=True),
    ],
    ids=["decoding", "prefill_decoding", "dms_decoding"],
)
def test_pipeline_rejects_multiple_questions_with_decoding_compression(
    kv_press_unit_test_pipeline, make_press  # noqa: F811
):
    with pytest.raises(ValueError, match="not compatible with multiple questions"):
        kv_press_unit_test_pipeline(CONTEXT, questions=QUESTIONS, press=make_press(), max_new_tokens=3)


@pytest.mark.parametrize(
    "make_press",
    [
        lambda: PrefillDecodingPress(prefilling_press=KnormPress(compression_ratio=0.5)),
        lambda: DMSPress(press=RandomPress(), threshold=0.5, sliding_window_size=0),
    ],
    ids=["prefill_decoding", "dms_prefill"],
)
def test_pipeline_accepts_multiple_questions_with_prefill_compression(
    kv_press_unit_test_pipeline, make_press  # noqa: F811
):
    answers = kv_press_unit_test_pipeline(CONTEXT, questions=QUESTIONS, press=make_press(), max_new_tokens=3)
    assert len(answers["answers"]) == 2
