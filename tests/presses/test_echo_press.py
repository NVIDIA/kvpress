# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress import EchoPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_kvzip_plus_normalization_is_not_supported():
    with pytest.raises(AssertionError):
        EchoPress(kvzip_plus_normalization=True)


def test_quantile_map_interpolates_and_clips():
    source = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[10.0, 20.0, 30.0]])
    values = torch.tensor([[0.0, 1.5, 2.5, 3.0, 4.0]])
    mapped = EchoPress._quantile_map(source, target, values)
    torch.testing.assert_close(mapped, torch.tensor([[10.0, 15.0, 25.0, 30.0, 30.0]]))


def test_quantile_map_is_fitted_per_row():
    scales = torch.tensor([[1.0], [100.0]])
    source = torch.tensor([3.0, 1.0, 2.0]).expand(2, 3)
    target = source * scales
    values = torch.tensor([0.0, 1.5, 4.0]).expand(2, 3)
    mapped = EchoPress._quantile_map(source, target, values)
    torch.testing.assert_close(mapped, torch.tensor([1.0, 1.5, 3.0]) * scales)


@pytest.mark.parametrize("chunk_size", [64, 512])
def test_single_replay_pass_and_calibration(unit_test_model, monkeypatch, chunk_size):  # noqa: F811
    model = unit_test_model
    press = EchoPress(compression_ratio=0.5, chunk_size=chunk_size)
    n_tokens = 300
    ids = torch.randint(3, model.config.vocab_size, (1, n_tokens), device=model.device)

    forward_lengths = []
    hook = model.model.register_forward_pre_hook(
        lambda module, args, kwargs: forward_lengths.append(kwargs["input_ids"].shape[1]), with_kwargs=True
    )
    scores = []
    compress_post = press.compress_post
    monkeypatch.setattr(press, "compress_post", lambda m: (scores.append(press.score_val.clone()), compress_post(m)))

    cache = DynamicCache()
    try:
        with torch.no_grad(), press(model):
            model(input_ids=ids, past_key_values=cache, logits_to_keep=1)
    finally:
        hook.remove()

    # One prefill pass, then a single replay pass over the first chunk and its repeat prompt
    assert len(forward_lengths) == 2
    assert forward_lengths[0] == n_tokens
    assert min(chunk_size, n_tokens) < forward_lengths[1] < min(chunk_size, n_tokens) + 32

    # KVzip-style eviction masks keys instead of removing them
    n_masked = sum(len(layer.self_attn.masked_key_indices[0]) for layer in model.model.layers)
    n_total = model.config.num_hidden_layers * model.config.num_key_value_heads * n_tokens
    assert n_masked == int(n_total * press.compression_ratio)
    assert cache.get_seq_length() == n_tokens

    # Later chunks are mapped onto the range of the exact first-chunk scores, per layer and KV head
    (score_val,) = scores
    assert torch.isfinite(score_val).all()
    first_chunk, later_chunks = score_val[..., :chunk_size], score_val[..., chunk_size:]
    if later_chunks.shape[-1] > 0:
        assert (later_chunks >= first_chunk.amin(-1, keepdim=True)).all()
        assert (later_chunks <= first_chunk.amax(-1, keepdim=True)).all()
