# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import numpy as np
import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import kvpress.presses.duo_attention_press as duo


@pytest.fixture
def local_samples(monkeypatch):
    texts = ["one two three four five six seven", "seven three one six four two five one"]
    words = ["[UNK]", "one", "two", "three", "four", "five", "six", "seven"]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", model_input_names=["input_ids"])
    dataset = Dataset.from_dict({"chapter": texts})
    # Replace only network-backed loading, retaining real tokenization/model execution.
    monkeypatch.setattr(duo.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(duo, "load_dataset", lambda *args, **kwargs: dataset)
    return tokenizer, texts


def tiny_model(config_cls, model_cls, **config_kwargs):
    config = config_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        **config_kwargs,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return model_cls(config).cpu().eval()


@pytest.mark.parametrize(
    "config_cls, model_cls, native_heads_first",
    [(Qwen3Config, Qwen3ForCausalLM, False), (Gemma3TextConfig, Gemma3ForCausalLM, True)],
)
def test_on_the_fly_replays_native_qk_normalization(config_cls, model_cls, native_heads_first, local_samples):
    model = tiny_model(config_cls, model_cls)
    attention = model.model.layers[0].self_attn
    queries, keys = [], []
    with ExitStack() as stack:
        for norm, captured in [(attention.q_norm, queries), (attention.k_norm, keys)]:
            handle = norm.register_forward_hook(lambda module, args, output, captured=captured: captured.append(output))
            stack.callback(handle.remove)
        duo.duo_attention_on_the_fly.__wrapped__(model, num_samples=1, q_len=16)

    for captured in [queries, keys]:
        # One native forward call and one reconstruction call, before averaging.
        assert len(captured) == 2
        native, reconstructed = captured
        if native_heads_first:
            native = native.transpose(1, 2)
        torch.testing.assert_close(reconstructed, native)


@pytest.mark.parametrize(
    "config_cls, model_cls, config_kwargs",
    [
        (Qwen3Config, Qwen3ForCausalLM, {}),
        (LlamaConfig, LlamaForCausalLM, {}),
        (Gemma3TextConfig, Gemma3ForCausalLM, {"layer_types": ["sliding_attention", "full_attention"]}),
        (Gemma3TextConfig, Gemma3ForCausalLM, {"layer_types": ["full_attention", "sliding_attention"]}),
    ],
)
def test_on_the_fly_scores_match_native_attention_states(config_cls, model_cls, config_kwargs, local_samples):
    tokenizer, texts = local_samples
    model = tiny_model(config_cls, model_cls, **config_kwargs)
    q_len = 16
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    num_groups = num_heads // num_kv_heads
    queries = [[] for _ in model.model.layers]
    keys = [[] for _ in model.model.layers]

    # Capture the actual per-layer RoPE tensors from a native forward at the
    # scorer's synthetic length, without duplicating its API-selection logic.
    positions = [[] for _ in model.model.layers]
    with ExitStack() as stack, torch.no_grad():
        for layer, captured in zip(model.model.layers, positions):
            handle = layer.self_attn.register_forward_pre_hook(
                lambda module, args, kwargs, captured=captured: captured.append(kwargs["position_embeddings"]),
                with_kwargs=True,
            )
            stack.callback(handle.remove)
        model(input_ids=torch.ones((1, q_len), dtype=torch.long))
    if config_cls is Gemma3TextConfig:
        assert not torch.allclose(positions[0][0][0], positions[1][0][0])

    with ExitStack() as stack, torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            attention = layer.self_attn
            q_module = getattr(attention, "q_norm", attention.q_proj)
            k_module = getattr(attention, "k_norm", attention.k_proj)
            for module, captured in [(q_module, queries[i]), (k_module, keys[i])]:
                handle = module.register_forward_hook(
                    lambda module, args, output, captured=captured: captured.append(output.detach())
                )
                stack.callback(handle.remove)
        for text in texts:
            model(**tokenizer(text, return_tensors="pt"))

    expected = []
    with torch.no_grad():
        for layer_queries, layer_keys, [(cos, sin)] in zip(queries, keys, positions):
            samples = []
            for q, k in zip(layer_queries, layer_keys):
                if config_cls is Gemma3TextConfig:
                    q, k = q.transpose(1, 2), k.transpose(1, 2)
                q = q.reshape(1, -1, num_heads, head_dim).transpose(1, 2).mean(dim=2, keepdim=True)
                k = k.reshape(1, -1, num_kv_heads, head_dim).transpose(1, 2).mean(dim=2, keepdim=True)
                q, k = q.expand(-1, -1, q_len, -1), k.expand(-1, -1, q_len, -1)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k = k.repeat_interleave(num_groups, dim=1)
                probabilities = (q[:, :, -1:] @ k.transpose(-1, -2) / head_dim**0.5).softmax(-1)
                # Each probability contributes to all cumulative sums after it.
                weights = torch.arange(q_len, 0, -1) / q_len
                areas = (probabilities * weights).sum(-1).reshape(num_kv_heads, num_groups).mean(-1)
                samples.append(areas)
            expected.append(torch.stack(samples).mean(0))
    expected = torch.stack(expected).numpy()
    actual = duo.duo_attention_on_the_fly.__wrapped__(model, num_samples=len(texts), q_len=q_len)
    assert actual.shape == (2, 2)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
