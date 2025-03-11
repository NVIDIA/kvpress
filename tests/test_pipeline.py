# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

import pytest
import torch
from transformers import AutoTokenizer, DynamicCache, QuantizedCacheConfig, QuantoQuantizedCache
from transformers.utils import is_optimum_quanto_available

from kvpress import ExpectedAttentionPress
from kvpress.pipeline import KVPressTextGenerationPipeline
from tests.fixtures import danube_500m_model  # noqa: F401
from tests.fixtures import kv_press_danube_pipeline  # noqa: F401
from tests.fixtures import kv_press_unit_test_pipeline  # noqa: F401
from tests.fixtures import unit_test_model  # noqa: F401


def test_pipeline(kv_press_unit_test_pipeline, caplog):  # noqa: F811
    with caplog.at_level(logging.DEBUG):
        context = "This is a test article. It was written on 2022-01-01."
        questions = ["When was this article written?"]
        press = ExpectedAttentionPress(compression_ratio=0.4)
        answers = kv_press_unit_test_pipeline(context, questions=questions, press=press)["answers"]

    assert len(answers) == 1
    assert isinstance(answers[0], str)

    messages = [record.message for record in caplog.records]
    assert "Context Length: 23" in messages, messages
    assert "Compressed Context Length: 13" in messages, messages


def test_pipeline_with_cache(kv_press_unit_test_pipeline, caplog):  # noqa: F811
    context = "This is a test article. It was written on 2022-01-01."
    questions = ["When was this article written?"]
    press = ExpectedAttentionPress(compression_ratio=0.4)
    cache = DynamicCache()
    answers = kv_press_unit_test_pipeline(context, questions=questions, press=press, cache=cache)["answers"]

    assert len(answers) == 1
    assert isinstance(answers[0], str)


@pytest.mark.parametrize("question", ["When was this article written?", ""])
def test_pipeline_single_or_no_question(kv_press_unit_test_pipeline, question, caplog):  # noqa: F811
    with caplog.at_level(logging.DEBUG):
        context = "This is a test article. It was written on 2022-01-01."
        press = ExpectedAttentionPress(compression_ratio=0.4)
        answer = kv_press_unit_test_pipeline(context, question=question, press=press)["answer"]

    assert isinstance(answer, str)

    messages = [record.message for record in caplog.records]
    assert "Context Length: 23" in messages, messages
    assert "Compressed Context Length: 13" in messages, messages


def test_pipeline_no_press_works(kv_press_unit_test_pipeline, caplog):  # noqa: F811
    context = "This is a test article. It was written on 2022-01-01."
    question = "When was this article written?"
    kv_press_unit_test_pipeline(context, question=question)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_pipeline_answer_is_correct(danube_500m_model, caplog):  # noqa: F811
    with caplog.at_level(logging.DEBUG):
        answers = generate_answer(danube_500m_model)

    for answer in answers:
        assert answer == "This article was written on January 1, 2022."

    messages = [record.message for record in caplog.records]
    assert "Context Length: 28" in messages
    assert "Compressed Context Length: 16" in messages


@pytest.mark.skipif(not is_optimum_quanto_available(), reason="Optimum Quanto is not available")
def test_pipeline_with_quantized_cache(kv_press_danube_pipeline, caplog):  # noqa: F811
    with caplog.at_level(logging.DEBUG):
        context = "This is a test article. It was written on 2022-01-01."
        questions = ["When was this article written?"]
        press = ExpectedAttentionPress(compression_ratio=0.4)
        config = QuantizedCacheConfig(nbits=4)
        cache = QuantoQuantizedCache(config)
        answers = kv_press_danube_pipeline(context, questions=questions, press=press, cache=cache)["answers"]

    assert len(answers) == 1
    assert isinstance(answers[0], str)

    for answer in answers:
        assert answer == "This article was written on January 1, 2022."

    messages = [record.message for record in caplog.records]
    assert "Context Length: 28" in messages
    assert "Compressed Context Length: 16" in messages


def test_pipeline_compresses_context(unit_test_model, caplog):  # noqa: F811
    with caplog.at_level(logging.DEBUG):
        answers = generate_answer(unit_test_model)

    assert len(answers) == 2
    assert isinstance(answers[0], str)

    messages = [record.message for record in caplog.records]
    assert "Context Length: 23" in messages, messages
    assert "Compressed Context Length: 13" in messages, messages


@torch.no_grad()
def test_pipeline_context_cache_is_invariant(unit_test_model):  # noqa: F811
    model = unit_test_model
    questions = ["When was this article written?"]
    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    compression_pipeline = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)
    input_ids_question = tokenizer(questions[0], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    seq_len = 256
    past_key_values: DynamicCache = model(
        input_ids=torch.randint(0, 1000, (1, seq_len), device=device), past_key_values=DynamicCache()
    ).past_key_values
    assert past_key_values.get_seq_length() == seq_len

    keys = [key.clone() for key in past_key_values.key_cache]
    values = [value.clone() for value in past_key_values.value_cache]
    compression_pipeline.generate_answer(input_ids_question, past_key_values, context_length=22, max_new_tokens=10)
    assert past_key_values.get_seq_length() == seq_len
    assert all([torch.allclose(key, new_key) for key, new_key in zip(keys, past_key_values.key_cache)])
    assert all([torch.allclose(value, new_value) for value, new_value in zip(values, past_key_values.value_cache)])


def generate_answer(model):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    context = "This is a test article. It was written on 2022-01-01."
    questions = ["When was this article written?", "When was this article written?"]
    press = ExpectedAttentionPress(compression_ratio=0.4)
    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    answers = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)(
        context, questions=questions, press=press
    )["answers"]
    model.to(torch.device("cpu"))
    return answers
