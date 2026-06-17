# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import kvpress.presses.kvzip_press as kvzip_press_module
from kvpress import KVzipPress


class FakeTokenizer:
    chat_template = None

    def encode(self, text, return_tensors=None, add_special_tokens=False):
        return torch.tensor([[0]])


class FakeInnerModel:
    layers = []
    rotary_emb = None

    def forward(self, *args, **kwargs):
        return None


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(name_or_path="fake-model")
        self.model = FakeInnerModel()


class RecordingKVzipPress(KVzipPress):
    def _perform_kvzip_compression(self, model, tokenizer):
        self.forward_during_compression = model.model.forward


def test_kvzip_press_restores_model_forward_if_context_raises(monkeypatch):
    monkeypatch.setattr(kvzip_press_module.AutoTokenizer, "from_pretrained", lambda _: FakeTokenizer())
    model = FakeModel()
    original_forward = model.model.forward
    press = KVzipPress(compression_ratio=0)

    with pytest.raises(RuntimeError):
        with press(model):
            assert model.model.forward != original_forward
            raise RuntimeError("boom")

    assert model.model.forward == original_forward


def test_kvzip_press_restores_model_forward_before_compression(monkeypatch):
    monkeypatch.setattr(kvzip_press_module.AutoTokenizer, "from_pretrained", lambda _: FakeTokenizer())
    model = FakeModel()
    original_forward = model.model.forward
    press = RecordingKVzipPress(compression_ratio=0.5)

    with press(model):
        model.model.forward(input_ids=torch.tensor([[1]]), past_key_values=object())

    assert press.forward_during_compression == original_forward
    assert model.model.forward == original_forward
