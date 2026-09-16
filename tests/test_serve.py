# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kvpress import KnormPress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.serve import (
    apply_chat_template,
    chat_completion_response,
    configure_press,
    create_app,
    default_press_registry,
    enable_thinking_from_body,
    message_content_to_text,
    parse_args,
)
from tests.fixtures import unit_test_model  # noqa: F401


def test_press_registry_includes_eval_names():
    registry = default_press_registry()
    for name in ("knorm", "snapkv", "kvzip", "no_press", "decoding_knorm"):
        assert name in registry
    assert registry["no_press"] is None


def test_configure_press_sets_compression_ratio():
    press = configure_press(KnormPress(), compression_ratio=0.4)
    assert press.compression_ratio == 0.4


def test_configure_decoding_press_target_size():
    press = configure_press(DecodingPress(base_press=KnormPress()), target_size=128, compression_interval=10)
    assert press.target_size == 128
    assert press.compression_interval == 10


def test_message_content_parts_and_thinking_body():
    assert message_content_to_text([{"type": "text", "text": "hello"}]) == "hello"
    assert enable_thinking_from_body({"chat_template_kwargs": {"enable_thinking": True}}, False) is True
    assert (
        enable_thinking_from_body({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}, True) is False
    )
    assert enable_thinking_from_body({}, True) is True


def test_parse_args_press_and_port():
    args = parse_args(["--model", "Qwen/Qwen3-4B", "--press", "knorm", "--compression-ratio", "0.5", "--port", "8002"])
    assert args.model == "Qwen/Qwen3-4B"
    assert args.press == "knorm"
    assert args.compression_ratio == 0.5
    assert args.port == 8002


def test_chat_completion_response_shape():
    from kvpress.serve import GenerationResult

    payload = chat_completion_response(
        "unit",
        GenerationResult(text="ok", prompt_tokens=3, completion_tokens=2, finish_reason="stop"),
    )
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "ok"
    assert payload["usage"]["total_tokens"] == 5


def test_openai_endpoints_with_unit_model(unit_test_model):  # noqa: F811
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from transformers import AutoTokenizer

    from kvpress.serve import KVPressEngine

    tokenizer = AutoTokenizer.from_pretrained("MaxJeblick/llama2-0b-unit-test")
    engine = KVPressEngine(
        model=unit_test_model,
        tokenizer=tokenizer,
        model_id="unit-test",
        press=KnormPress(compression_ratio=0.2),
    )
    prompt = apply_chat_template(tokenizer, [{"role": "user", "content": "hi"}], enable_thinking=False)
    assert "hi" in prompt

    client = TestClient(create_app(engine))
    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "unit-test"

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "unit-test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] > 0

    streamed = client.post(
        "/v1/chat/completions",
        json={"model": "unit-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert streamed.status_code == 400
