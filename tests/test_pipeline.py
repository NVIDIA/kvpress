# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

import pytest
import torch
from transformers import AutoTokenizer, DynamicCache, QuantizedCache
from transformers.utils import is_flash_attn_2_available, is_optimum_quanto_available

from kvpress import ExpectedAttentionPress, KVzipPress
from kvpress.pipeline import KVPressTextGenerationPipeline
from tests.fixtures import danube_500m_model  # noqa: F401
from tests.fixtures import kv_press_danube_pipeline  # noqa: F401
from tests.fixtures import unit_test_model  # noqa: F401
from tests.fixtures import kv_press_llama3_2_flash_attn_pipeline, kv_press_unit_test_pipeline  # noqa: F401


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


def test_pipeline_with_cache(kv_press_unit_test_pipeline):  # noqa: F811
    context = "This is a test article. It was written on 2022-01-01."
    questions = ["When was this article written?"]
    press = ExpectedAttentionPress(compression_ratio=0.4)
    cache = DynamicCache()
    answers = kv_press_unit_test_pipeline(context, questions=questions, press=press, cache=cache)["answers"]

    assert len(answers) == 1
    assert isinstance(answers[0], str)


class TestPipelineFA2:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
    @pytest.mark.skipif(not is_flash_attn_2_available(), reason="flash_attn is not installed")
    @pytest.mark.parametrize("compression_ratio", [0.0, 0.2])
    @pytest.mark.xfail(reason="Known issue (https://github.com/huggingface/transformers/issues/42550)", strict=False)
    def test_pipeline_fa2(self, kv_press_llama3_2_flash_attn_pipeline, compression_ratio):  # noqa: F811
        context = "This is a test article. It was written on 2022-01-01."
        questions = ["Repeat the last sentence"]
        press = ExpectedAttentionPress(compression_ratio=compression_ratio)
        cache = DynamicCache()
        answers = kv_press_llama3_2_flash_attn_pipeline(
            context, questions=questions, press=press, cache=cache, max_new_tokens=6
        )["answers"]

        assert len(answers) == 1
        assert isinstance(answers[0], str)

        kv_press_llama3_2_flash_attn_pipeline.model.set_attn_implementation("sdpa")
        press = ExpectedAttentionPress(compression_ratio=compression_ratio)
        cache = DynamicCache()
        answers_sdpa = kv_press_llama3_2_flash_attn_pipeline(
            context, questions=questions, press=press, cache=cache, max_new_tokens=6
        )["answers"]
        kv_press_llama3_2_flash_attn_pipeline.model.set_attn_implementation("flash_attention_2")

        assert (
            answers_sdpa[0] == answers[0]
        ), f"Answers from SDPA and Flash Attention 2 should be the same. \n{answers_sdpa[0]}\n{answers[0]}"
        assert "This is a test" in answers[0], f"The answer should contain the context sentence, but got {answers[0]}."


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
        cache = QuantizedCache(backend="quanto", config=kv_press_danube_pipeline.model.config, nbits=4)
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
    device = model.device

    compression_pipeline = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)
    input_ids_question = tokenizer(questions[0], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    seq_len = 256
    past_key_values: DynamicCache = model(
        input_ids=torch.randint(0, 1000, (1, seq_len), device=device), past_key_values=DynamicCache()
    ).past_key_values
    assert past_key_values.get_seq_length() == seq_len

    keys = [layer.keys.clone() for layer in past_key_values.layers]
    values = [layer.values.clone() for layer in past_key_values.layers]
    cache_seq_lengths = [past_key_values.get_seq_length(layer_idx) for layer_idx in range(len(past_key_values))]
    compression_pipeline.generate_answer(input_ids_question, past_key_values, context_length=22, max_new_tokens=10)
    compression_pipeline._remove_answer_from_cache(past_key_values, cache_seq_lengths)
    assert past_key_values.get_seq_length() == seq_len
    assert all([torch.allclose(key, layer.keys) for key, layer in zip(keys, past_key_values.layers)])
    assert all([torch.allclose(value, layer.values) for value, layer in zip(values, past_key_values.layers)])


def generate_answer(model):
    device = model.device
    context = "This is a test article. It was written on 2022-01-01."
    questions = ["When was this article written?", "When was this article written?"]
    press = ExpectedAttentionPress(compression_ratio=0.4)
    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    answers = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)(
        context, questions=questions, press=press
    )["answers"]
    return answers


@pytest.mark.parametrize("press_cls", [ExpectedAttentionPress, KVzipPress])
def test_pipeline_only_computes_the_logits_it_needs(kv_press_unit_test_pipeline, press_cls):  # noqa: F811
    """
    Both the question forward pass in ``_generate_answer`` and the context reconstruction forward
    passes in ``KVzipPress`` must only materialize the logits of the last position. Requesting this
    under the transformers name that was renamed to ``logits_to_keep`` is swallowed by ``**kwargs``
    without any error, and the full ``(1, seq_len, vocab_size)`` logits tensor is computed instead.
    """
    model = kv_press_unit_test_pipeline.model
    original_forward = model.forward
    logits_lengths = []

    def recording_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        logits_lengths.append(outputs.logits.shape[1])
        return outputs

    context = "This is a test article. It was written on 2022-01-01."
    questions = ["When was this article written?"]

    model.forward = recording_forward
    try:
        kv_press_unit_test_pipeline(
            context, questions=questions, press=press_cls(compression_ratio=0.4), max_new_tokens=2
        )
    finally:
        model.forward = original_forward

    assert logits_lengths, "no forward pass through the lm_head was recorded"
    assert all(length == 1 for length in logits_lengths), logits_lengths
