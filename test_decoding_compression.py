#!/usr/bin/env python3

"""
Test script to verify that DecodingPress actually compresses during decoding.
"""
import logging

import pytest
import torch
from transformers import DynamicCache, pipeline

from kvpress import ScorerPress, SnapKVPress, QFilterPress, PyramidKVPress
from kvpress.presses.generation.decoding_press import DecodingPress
from kvpress.presses.generation.prefill_decoding_press import PrefillDecodingPress
from kvpress.presses.knorm_press import KnormPress
from tests.default_presses import default_presses

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("token_buffer_size", [32, 64, 128])
def test_decoding_compression(token_buffer_size):
    """Test that DecodingPress compresses the cache during decoding."""

    # Initialize pipeline with a small model
    pipe = pipeline("kv-press-text-generation",
                    model="MaxJeblick/llama2-0b-unit-test",
                    device_map="auto")

    # Create a DecodingPress with KnormPress
    press = DecodingPress(
        base_press=KnormPress(compression_ratio=0.5),  # Remove 50% of tokens
        compression_steps=4,  # Compress every 4 tokens
        token_buffer_size=token_buffer_size
    )

    # Create cache
    cache = DynamicCache()

    # Test context and question
    context = "The quick brown fox jumps over the lazy dog. " * 10  # Repeat for longer context
    question = "What animal jumps over the dog?"

    # Run pipeline
    result = pipe(
        context,
        question=question,
        press=press,
        cache=cache,
        max_new_tokens=20
    )

    # Assert that all layers have the expected cache size
    for layer_idx, key_tensor in enumerate(cache.key_cache):
        layer_seq_len = key_tensor.shape[2]
        assert layer_seq_len == token_buffer_size, (
            f"Layer {layer_idx}: Expected cache sequence length to be {token_buffer_size}, "
            f"but got {layer_seq_len}"
        )


def test_prefill_decoding_press_calls_both_phases():
    """Test that PrefillDecodingPress calls both prefilling and decoding presses."""

    # Initialize pipeline
    pipe = pipeline("kv-press-text-generation",
                    model="MaxJeblick/llama2-0b-unit-test",
                    device_map="auto")

    # Create PrefillDecodingPress with both presses
    combined_press = PrefillDecodingPress(
        prefilling_press=KnormPress(compression_ratio=0.6),  # Compress to 60% during prefill
        decoding_press=DecodingPress(
            base_press=KnormPress(),
            compression_steps=3,
            token_buffer_size=48
        )
    )

    # Test context and question
    context = "The quick brown fox jumps over the lazy dog. " * 12  # Longer context
    question = "What animal jumps over the dog?"

    # Run pipeline
    cache = DynamicCache()
    result = pipe(context, question=question, press=combined_press, cache=cache, max_new_tokens=15)

    # Check that cache was compressed during both phases
    # Final cache should be compressed to decoding press target size
    for layer_idx, key_tensor in enumerate(cache.key_cache):
        layer_seq_len = key_tensor.shape[2]
        assert layer_seq_len == 48, (  # token_buffer_size from decoding press
            f"Layer {layer_idx}: Expected final cache size to be 48 (decoding target), "
            f"but got {layer_seq_len}"
        )


def test_decoding_press_without_prefill():
    """Test that DecodingPress works correctly when used standalone (no prefill compression)."""

    # Initialize pipeline
    pipe = pipeline("kv-press-text-generation",
                    model="MaxJeblick/llama2-0b-unit-test",
                    device_map="auto")

    # Create DecodingPress only
    decoding_press = DecodingPress(
        base_press=KnormPress(compression_ratio=0.4),
        compression_steps=5,
        token_buffer_size=64
    )

    # Test context and question
    context = "The quick brown fox jumps over the lazy dog. " * 8
    question = "What animal jumps over the dog?"

    # Run pipeline
    cache = DynamicCache()
    result = pipe(context, question=question, press=decoding_press, cache=cache, max_new_tokens=25)

    # Check that cache was compressed during decoding
    for layer_idx, key_tensor in enumerate(cache.key_cache):
        layer_seq_len = key_tensor.shape[2]
        assert layer_seq_len == 64, (
            f"Layer {layer_idx}: Expected cache size to be 64, but got {layer_seq_len}"
        )


def test_prefill_decoding_press_decoding_only():
    """Test PrefillDecodingPress with only decoding press (no prefill compression)."""

    # Initialize pipeline
    pipe = pipeline("kv-press-text-generation",
                    model="MaxJeblick/llama2-0b-unit-test",
                    device_map="auto")

    # Create PrefillDecodingPress with only decoding press
    combined_press = PrefillDecodingPress(
        prefilling_press=None,
        decoding_press=DecodingPress(
            base_press=KnormPress(compression_ratio=0.6),
            compression_steps=4,
            token_buffer_size=56
        )
    )

    # Test context and question
    context = "The quick brown fox jumps over the lazy dog. " * 9
    question = "What animal jumps over the dog?"

    # Run pipeline
    cache = DynamicCache()
    result = pipe(context, question=question, press=combined_press, cache=cache, max_new_tokens=12)

    # Check that only decoding compression was applied
    for layer_idx, key_tensor in enumerate(cache.key_cache):
        layer_seq_len = key_tensor.shape[2]
        assert layer_seq_len == 56, (
            f"Layer {layer_idx}: Expected cache size to be 56, but got {layer_seq_len}"
        )


def test_decoding_press_equivalence():
    """Test that DecodingPress standalone yields same result as PrefillDecodingPress with decoding only."""

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize pipeline
    pipe = pipeline("kv-press-text-generation",
                    model="MaxJeblick/llama2-0b-unit-test",
                    device_map="auto")

    # Create standalone decoding press
    decoding_press = DecodingPress(
        base_press=KnormPress(compression_ratio=0.5),
        compression_steps=3,
        token_buffer_size=52
    )

    # Create PrefillDecodingPress with only decoding press
    combined_press = PrefillDecodingPress(
        prefilling_press=None,
        decoding_press=DecodingPress(
            base_press=KnormPress(compression_ratio=0.5),
            compression_steps=3,
            token_buffer_size=52
        )
    )

    # Test context and question
    context = "The quick brown fox jumps over the lazy dog. " * 7
    question = "What animal jumps over the dog?"

    # Run with standalone decoding press
    cache1 = DynamicCache()
    result1 = pipe(context, question=question, press=decoding_press, cache=cache1, max_new_tokens=10)

    # Run with combined press (decoding only)
    cache2 = DynamicCache()
    result2 = pipe(context, question=question, press=combined_press, cache=cache2, max_new_tokens=10)

    # Compare cache sizes (should be identical)
    for layer_idx in range(len(cache1.key_cache)):
        cache1_size = cache1.key_cache[layer_idx].shape[2]
        cache2_size = cache2.key_cache[layer_idx].shape[2]
        assert cache1_size == cache2_size, (
            f"Layer {layer_idx}: Standalone decoding cache size {cache1_size} != "
            f"combined press cache size {cache2_size}"
        )

    # Compare generated text results (should be identical)
    assert result1['answer'] == result2['answer'], (
        f"Generated answers differ:\n"
        f"Standalone decoding: '{result1['answer']}'\n"
        f"Combined press: '{result2['answer']}'"
    )

"""
E       AttributeError: 'QFilterPress' object has no attribute 'q_filters'
E           Failed: DecodingPress failed with SnapKVPress: shape '[1, 2, 2, 6]' is invalid for input of size 12
>       query_states = query_states.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)
E       RuntimeError: shape '[1, 2, 2, 6]' is invalid for input of size 12
"""
@pytest.mark.parametrize("press_config", default_presses)
def test_all_presses_work_with_decoding_press(press_config):
    """Test that all default presses work as base presses for DecodingPress."""

    # Initialize pipeline
    pipe = pipeline("kv-press-text-generation",
                    model="MaxJeblick/llama2-0b-unit-test",
                    device_map="auto")

    # Get press class and use the first (easier) configuration
    press_cls = press_config["cls"]
    press_kwargs = press_config["kwargs"][0]  # Use easier compression settings

    base_press = press_cls(**press_kwargs)
    if not isinstance(base_press, ScorerPress):
        logger.info(f"Press {press_cls.__name__} is not a ScorerPress, skipping test")
        return
    if isinstance(base_press, (SnapKVPress, QFilterPress, PyramidKVPress)):
        # Qfilter: __post_init__ not called
        # snapkv: hidden_states[:, -window_size:] may be an issue?
        #
        logger.info(f"Press {press_cls.__name__} is not supported, skipping test")
        #return

    # Create DecodingPress with this base press
    decoding_press = DecodingPress(
        base_press=base_press,
        compression_steps=3,
        token_buffer_size=48
    )

    # Test context and question
    context = "The quick brown fox jumps over the lazy dog. " * 8
    question = "What animal jumps over the dog?"

    # Run pipeline
    cache = DynamicCache()
    try:
        result = pipe(
            context,
            question=question,
            press=decoding_press,
            cache=cache,
            max_new_tokens=15
        )

        # Verify compression worked
        assert len(result['answer']) > 0, f"No answer generated with {press_cls.__name__}"

        # Check that cache was compressed (allow some tolerance for rounding)
        for layer_idx, key_tensor in enumerate(cache.key_cache):
            layer_seq_len = key_tensor.shape[2]
            assert 45 <= layer_seq_len <= 50, (
                f"{press_cls.__name__}: Layer {layer_idx} cache size {layer_seq_len} not in expected range [45-50]"
            )

    except Exception as e:
        pytest.fail(f"DecodingPress failed with {press_cls.__name__}: {e}")


def test_compression_actually_reduces_memory():
    """Test that compression actually reduces memory usage compared to no compression."""

    pipe = pipeline("kv-press-text-generation",
                    model="MaxJeblick/llama2-0b-unit-test",
                    device_map="auto")

    context = "The quick brown fox jumps over the lazy dog. " * 15  # Long context
    question = "What animal jumps over the dog?"

    # Run without compression
    cache_uncompressed = DynamicCache()
    result_uncompressed = pipe(context, question=question, cache=cache_uncompressed, max_new_tokens=25)

    # Run with compression
    press = DecodingPress(
        base_press=KnormPress(compression_ratio=0.3),  # Aggressive compression
        compression_steps=3,
        token_buffer_size=40
    )
    cache_compressed = DynamicCache()
    result_compressed = pipe(context, question=question, press=press, cache=cache_compressed, max_new_tokens=25)

    # Calculate memory usage (approximate)
    uncompressed_memory = sum(tensor.numel() * tensor.element_size()
                              for tensor in cache_uncompressed.key_cache + cache_uncompressed.value_cache)
    compressed_memory = sum(tensor.numel() * tensor.element_size()
                            for tensor in cache_compressed.key_cache + cache_compressed.value_cache)

    # Compression should significantly reduce memory usage
    compression_ratio = compressed_memory / uncompressed_memory
    assert compression_ratio < 0.6, (
        f"Expected compression ratio < 0.6, but got {compression_ratio:.3f} "
        f"(compressed: {compressed_memory} bytes, uncompressed: {uncompressed_memory} bytes)"
    )

    # Both should still generate reasonable answers
    assert len(result_uncompressed['answer']) > 0
    assert len(result_compressed['answer']) > 0
