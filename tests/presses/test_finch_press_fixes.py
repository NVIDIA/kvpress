# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest
import torch
from transformers import (
    AutoTokenizer,
    DynamicCache,
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
)

from kvpress import FinchPress
from kvpress.utils import compute_n_kept
from tests.fixtures import unit_test_model  # noqa: F401


def make_module(num_heads):
    return types.SimpleNamespace(head_dim=1, config=types.SimpleNamespace(num_attention_heads=num_heads))


@pytest.mark.parametrize("context_length, window_size, chunk_length", [(10, 3, 5), (12, 4, 5), (7, 5, 3)])
def test_chunked_compression_keeps_the_window(context_length, window_size, chunk_length):
    torch.manual_seed(0)
    k_len = context_length + window_size
    press = FinchPress(compression_ratio=0.5, chunk_length=chunk_length, rerotate_keys=False)
    press.window_size = window_size
    keys = torch.arange(k_len, dtype=torch.float32).view(1, 1, k_len, 1)
    attentions = torch.rand(1, 1, k_len, k_len).softmax(dim=-1)

    new_keys, _ = press.compress(make_module(1), None, keys, keys.clone(), attentions, {})

    kept = new_keys[0, 0, :, 0].long().tolist()
    assert set(range(context_length, k_len)) <= set(kept)
    chunk_lengths = [min(chunk_length, context_length - i) for i in range(0, context_length, chunk_length)]
    assert len(kept) == sum(compute_n_kept(length, 0.5) for length in chunk_lengths) + window_size


def test_normalized_scores_do_not_overflow_in_float16():
    k_len = 80_001
    press = FinchPress(compression_ratio=0.5)
    press.window_size = 1
    keys = torch.zeros(1, 1, k_len, 1, dtype=torch.float16)
    attentions = torch.full((1, 1, 1, k_len), 0.9, dtype=torch.float16)
    scores = press.score(make_module(1), None, keys, keys, attentions, {})
    assert torch.isfinite(scores).all()
    torch.testing.assert_close(scores[0, 0, 0], torch.tensor(0.9 * 80_000), rtol=1e-3, atol=0)


@torch.no_grad()
def test_window_size_is_not_reused_across_samples(unit_test_model):  # noqa: F811
    press = FinchPress(compression_ratio=0.5)
    press.delimiter_token_id = unit_test_model.config.eos_token_id
    input_ids = torch.arange(10, 30, device=unit_test_model.device)
    input_ids_with_delimiter = input_ids.clone()
    input_ids_with_delimiter[15] = press.delimiter_token_id
    with press(unit_test_model):
        unit_test_model(input_ids_with_delimiter.unsqueeze(0))
        with pytest.raises(AssertionError, match="window_size must be provided"):
            unit_test_model(input_ids.unsqueeze(0))


def test_update_model_and_tokenizer_does_not_shrink_embeddings():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    config = Qwen2Config(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1
    )
    vocab_size = config.vocab_size
    assert vocab_size > len(tokenizer) + 1
    model = Qwen2ForCausalLM(config)

    press = FinchPress()
    press.update_model_and_tokenizer(model, tokenizer)
    assert model.get_input_embeddings().num_embeddings == vocab_size
    assert press.delimiter_token_id == len(tokenizer) - 1


def test_update_model_and_tokenizer_grows_embeddings_when_needed():
    tokenizer = AutoTokenizer.from_pretrained("MaxJeblick/llama2-0b-unit-test")
    vocab_size = len(tokenizer)
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=vocab_size,
    )
    model = LlamaForCausalLM(config)

    press = FinchPress()
    press.update_model_and_tokenizer(model, tokenizer)
    assert model.get_input_embeddings().num_embeddings == len(tokenizer) == vocab_size + 1


@torch.no_grad()
def test_finch_press_with_gemma3_for_conditional_generation():
    text_config = Gemma3TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=300,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=8,
    )
    vision_config = SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=28, patch_size=14
    )
    config = Gemma3Config(
        text_config=text_config,
        vision_config=vision_config,
        mm_tokens_per_image=4,
        image_token_index=299,
        boi_token_index=297,
        eoi_token_index=298,
    )
    model = Gemma3ForConditionalGeneration(config).eval()
    assert not hasattr(model.model, "embed_tokens")

    press = FinchPress(compression_ratio=0.5, rerotate_keys=False)
    press.delimiter_token_id = 296
    input_ids = torch.arange(10, 30)
    input_ids[15] = press.delimiter_token_id
    cache = DynamicCache()
    with press(model):
        model(input_ids.unsqueeze(0), past_key_values=cache)
    # The delimiter is removed and only the full attention layer is compressed
    assert cache.layers[0].keys.shape[2] == 19
    assert cache.layers[1].keys.shape[2] == 9
