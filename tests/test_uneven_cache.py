# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from kvpress import KnormPress, PerLayerCompressionPress, PyramidKVPress, SnapKVPress
from kvpress.pipeline import KVPressTextGenerationPipeline


@pytest.fixture(params=["sdpa", "eager"])
def small_model(request):
    """Build a CPU model locally without downloading a checkpoint or tokenizer."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        attn_implementation=request.param,
        eos_token_id=31,
        pad_token_id=0,
    )
    return LlamaForCausalLM(config).eval()


@pytest.fixture(
    params=["pyramid", "per_layer", "increasing", "mixed", "empty_first", "empty_middle", "empty_last", "uniform"]
)
def press(request):
    """Exercise unequal layer budgets in both directions and a uniform-cache control."""
    if request.param == "pyramid":
        return PyramidKVPress(compression_ratio=0.5, window_size=4, beta=2)
    if request.param == "per_layer":
        return PerLayerCompressionPress(press=KnormPress(), compression_ratios=[0.25, 0.5, 0.75])
    if request.param == "increasing":
        return PerLayerCompressionPress(press=KnormPress(), compression_ratios=[0.75, 0.5, 0.25])
    if request.param == "mixed":
        return PerLayerCompressionPress(press=KnormPress(), compression_ratios=[0.5, 0.25, 0.75])
    if request.param == "empty_first":
        return PerLayerCompressionPress(press=KnormPress(), compression_ratios=[1.0, 0.5, 0.25])
    if request.param == "empty_middle":
        return PerLayerCompressionPress(press=KnormPress(), compression_ratios=[0.5, 1.0, 0.25])
    if request.param == "empty_last":
        return PerLayerCompressionPress(press=KnormPress(), compression_ratios=[0.5, 0.25, 1.0])
    return SnapKVPress(compression_ratio=0.5, window_size=4)


def context_ids():
    """Use enough context tokens to avoid PyramidKV's uniform-budget fallback."""
    return torch.arange(128).remainder(30).add(1).unsqueeze(0)


@torch.no_grad()
def test_pipeline_with_uneven_cache(small_model, press):
    """Question and answer generation must preserve every layer's compressed context."""
    vocabulary = {"[UNK]": 0, **{str(i): i for i in range(1, 32)}}
    tokenizer_object = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer_object.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object, unk_token="[UNK]")
    pipeline = KVPressTextGenerationPipeline(model=small_model, tokenizer=tokenizer, device="cpu")
    cache = DynamicCache()
    inputs = context_ids()
    with press(small_model):
        small_model.model(input_ids=inputs, past_key_values=cache)
    expected_lengths = [cache.get_seq_length(i) for i in range(3)]

    cache = DynamicCache()
    answers = pipeline._forward(
        {"context_ids": inputs, "questions_ids": [torch.tensor([[2, 3, 4, 5]]), torch.tensor([[6, 7, 8]])]},
        press=press,
        cache=cache,
        max_new_tokens=2,
    )

    assert len(answers) == 2
    assert all(isinstance(answer, str) for answer in answers)
    assert [cache.get_seq_length(i) for i in range(3)] == expected_lengths
    if isinstance(press, (PyramidKVPress, PerLayerCompressionPress)):
        assert len(set(expected_lengths)) == 3
        if isinstance(press, PerLayerCompressionPress) and 1.0 in press.compression_ratios:
            assert expected_lengths[press.compression_ratios.index(1.0)] == 0
    else:
        assert len(set(expected_lengths)) == 1


@torch.no_grad()
def test_generate_with_uneven_cache(small_model, press):
    """The standard Transformers generation API must also accept unequal layer budgets."""
    inputs = context_ids()
    with press(small_model):
        output = small_model.generate(inputs, max_new_tokens=3, do_sample=False)
    assert output.shape[1] > inputs.shape[1]


@torch.no_grad()
def test_uneven_cache_preserves_causality(small_model, press):
    """A multi-token question must match token-by-token decoding from the same cache."""
    cache = DynamicCache()
    inputs = context_ids()
    with press(small_model):
        small_model.model(input_ids=inputs, past_key_values=cache)
    query = torch.tensor([[2, 3, 4, 5]])
    positions = torch.arange(inputs.shape[1], inputs.shape[1] + query.shape[1]).unsqueeze(0)
    batched_logits = small_model(query, past_key_values=deepcopy(cache), position_ids=positions).logits
    sequential_logits = []
    for i in range(query.shape[1]):
        output = small_model(query[:, i : i + 1], past_key_values=cache, position_ids=positions[:, i : i + 1])
        sequential_logits.append(output.logits)

    torch.testing.assert_close(batched_logits, torch.cat(sequential_logits, dim=1), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("inference_mode", [False, True])
@pytest.mark.parametrize("compression_ratios", [[0.25, 0.5, 0.75], [0.75, 0.5, 0.25], [0.5, 0.25, 0.75]])
@torch.no_grad()
def test_shared_mask_is_validated_once_per_forward(small_model, monkeypatch, inference_mode, compression_ratios):
    """Scan a shared causal mask once per forward, including masks created in inference mode."""
    original_all = torch.Tensor.all
    scans = []
    mask_length = int(128 * (1 - compression_ratios[0])) + 4

    def count_mask_scans(tensor, *args, **kwargs):
        """Record full causal-mask reductions without changing their result."""
        if tensor.shape == (1, 1, 4, mask_length):
            scans.append(tensor.shape)
        return original_all(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "all", count_mask_scans)
    with torch.inference_mode(inference_mode), torch.no_grad():
        cache = DynamicCache()
        compression = PerLayerCompressionPress(KnormPress(), compression_ratios=compression_ratios)
        with compression(small_model):
            small_model.model(input_ids=context_ids(), past_key_values=cache)
        allowed = torch.arange(mask_length) <= torch.arange(4).unsqueeze(-1) + mask_length - 4
        mask = torch.zeros(1, 1, 4, mask_length).masked_fill(~allowed, -torch.inf)
        for expected_scans in [1, 2]:
            small_model(
                torch.tensor([[2, 3, 4, 5]]),
                past_key_values=deepcopy(cache),
                position_ids=torch.arange(128, 132).unsqueeze(0),
                attention_mask=mask,
            )
            assert len(scans) == expected_scans
