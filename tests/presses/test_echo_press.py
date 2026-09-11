# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress import EchoPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_default_is_per_request_per_head_virtual_to_exact():
    press = EchoPress(compression_ratio=0.5)
    assert press.exact_first_chunk
    assert press.score_calibration
    assert press.calibration_direction == "virtual_to_exact"
    assert press.calibration_scope == "head"


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("exact_to_virtual", [1.0, 2.0, 3.0, 1.5, 2.5, 3.0]),
        ("virtual_to_exact", [10.0, 20.0, 30.0, 15.0, 25.0, 30.0]),
    ],
)
def test_per_sample_calibration_direction(direction, expected):
    press = EchoPress(calibration_direction=direction)
    press.score_val = torch.tensor([[[[10.0, 20.0, 30.0, 1.5, 2.5, 3.0]]]])
    press._v0 = {0: torch.tensor([[1.0, 2.0, 3.0]])}
    press._chunks = [(0, 3, 0), (3, 6, 0)]

    press._apply_per_sample_calibration(0, 3)

    torch.testing.assert_close(press.score_val[0, 0, 0], torch.tensor(expected))


def test_online_calibration_separates_heads_and_clips_endpoints():
    press = EchoPress()
    scales = torch.tensor([[10.0, 100.0], [1000.0, 10000.0]])[..., None]
    virtual = torch.tensor([3.0, 1.0, 2.0]).expand(2, 2, 3)
    exact = virtual * scales
    later = torch.tensor([0.0, 1.5, 4.0]).expand(2, 2, 3)
    press.score_val = torch.cat((exact, later), dim=-1).unsqueeze(1)
    press._v0 = dict(enumerate(virtual))
    press._chunks = [(0, 3, 0), (3, 6, 0)]

    press._apply_per_sample_calibration(0, 3)

    torch.testing.assert_close(press.score_val[:, 0, :, :3], exact, rtol=0, atol=0)
    expected = torch.tensor([1.0, 1.5, 3.0]) * scales
    torch.testing.assert_close(press.score_val[:, 0, :, 3:], expected)


@pytest.mark.parametrize(
    ("exact_first_chunk", "score_calibration", "chunk_size"),
    [(True, True, 64), (True, False, 64), (False, True, 64), (True, True, 512)],
)
def test_online_scoring_across_requests(
    unit_test_model, monkeypatch, exact_first_chunk, score_calibration, chunk_size  # noqa: F811
):
    model = unit_test_model
    press = EchoPress(
        compression_ratio=0.5,
        chunk_size=chunk_size,
        exact_first_chunk=exact_first_chunk,
        score_calibration=score_calibration,
    )
    calibration_anchors = []
    calibrate = press._apply_per_sample_calibration

    def capture_calibration(start, end):
        exact = press.score_val[..., start:end].clone()
        calibration_anchors.append(press._v0[0].clone())
        calibrate(start, end)
        torch.testing.assert_close(press.score_val[..., start:end], exact, rtol=0, atol=0)
        assert torch.isfinite(press.score_val).all()

    monkeypatch.setattr(press, "_apply_per_sample_calibration", capture_calibration)
    generator = torch.Generator(device=model.device).manual_seed(42)
    for request_idx, n_tokens in enumerate((128, 300), start=1):
        ids = torch.randint(3, model.config.vocab_size, (1, n_tokens), device=model.device, generator=generator)
        forward_lengths = []

        def record_forward(module, args, kwargs):
            forward_lengths.append(kwargs["input_ids"].shape[1])

        hook = model.register_forward_pre_hook(record_forward, with_kwargs=True)
        cache = DynamicCache()
        try:
            with torch.no_grad(), press(model):
                model(input_ids=ids, past_key_values=cache, logits_to_keep=1)
        finally:
            hook.remove()

        assert forward_lengths[0] == n_tokens
        assert len(forward_lengths) == 1 + int(exact_first_chunk)
        if exact_first_chunk:
            assert forward_lengths[1] <= chunk_size
        should_calibrate = exact_first_chunk and score_calibration and chunk_size == 64
        assert len(calibration_anchors) == request_idx * int(should_calibrate)
        n_masked = sum(len(layer.self_attn.masked_key_indices[0]) for layer in model.model.layers)
        n_total = model.config.num_hidden_layers * model.config.num_key_value_heads * n_tokens
        assert n_masked == int(n_total * press.compression_ratio)
        assert cache.get_seq_length() == n_tokens  # KVzip-style eviction uses attention masks.
        assert not press._q and not press._v0

    if calibration_anchors:
        assert not torch.equal(calibration_anchors[0], calibration_anchors[1])
