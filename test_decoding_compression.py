#!/usr/bin/env python3

"""
Test script to verify that DecodingPress actually compresses during decoding.
"""

import pytest
import torch
from transformers import DynamicCache, pipeline

from kvpress.presses.generation.decoding_press import DecodingPress
from kvpress.presses.generation.prefill_decoding_press import PrefillDecodingPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.random_press import RandomPress


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


def test_prefill_decoding_press_prefill_only_equivalence():
    """Test that PrefillDecodingPress with only prefill press yields same result as standalone prefill press."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize pipeline
    pipe = pipeline("kv-press-text-generation",
                           model="MaxJeblick/llama2-0b-unit-test",
                           device_map="auto")
    
    # Create standalone prefill press
    prefill_press = RandomPress(compression_ratio=0.7, seed=123)
    
    # Create PrefillDecodingPress with only prefill press
    combined_press = PrefillDecodingPress(
        prefilling_press=RandomPress(compression_ratio=0.7, seed=123),
        decoding_press=None
    )
    
    # Test context and question
    context = "The quick brown fox jumps over the lazy dog. " * 8
    question = "What animal jumps over the dog?"
    
    # Run with standalone prefill press
    cache1 = DynamicCache()
    result1 = pipe(context, question=question, press=prefill_press, cache=cache1, max_new_tokens=10)
    
    # Run with combined press (prefill only)
    cache2 = DynamicCache()
    result2 = pipe(context, question=question, press=combined_press, cache=cache2, max_new_tokens=10)
    
    # Compare cache sizes after prefilling (should be identical)
    for layer_idx in range(len(cache1.key_cache)):
        cache1_size = cache1.key_cache[layer_idx].shape[2]
        cache2_size = cache2.key_cache[layer_idx].shape[2]
        assert cache1_size == cache2_size, (
            f"Layer {layer_idx}: Standalone prefill cache size {cache1_size} != "
            f"combined press cache size {cache2_size}"
        )
    
    # Compare generated text results (should be identical)
    assert result1['answer'] == result2['answer'], (
        f"Generated answers differ:\n"
        f"Standalone prefill: '{result1['answer']}'\n"
        f"Combined press: '{result2['answer']}'"
    )
    
    # Compare cache tensors directly (should be identical)
    for layer_idx in range(len(cache1.key_cache)):
        assert torch.allclose(cache1.key_cache[layer_idx], cache2.key_cache[layer_idx], atol=1e-6), (
            f"Layer {layer_idx}: Key cache tensors are not identical"
        )
        assert torch.allclose(cache1.value_cache[layer_idx], cache2.value_cache[layer_idx], atol=1e-6), (
            f"Layer {layer_idx}: Value cache tensors are not identical"
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


@pytest.mark.parametrize("compression_steps", [2, 5])
def test_decoding_compression_steps_parameter(compression_steps):
    """Test that DecodingPress respects different compression_steps values."""
    
    # Initialize pipeline
    pipe = pipeline("kv-press-text-generation",
                           model="MaxJeblick/llama2-0b-unit-test",
                           device_map="auto")
    
    # Create DecodingPress with different compression steps
    press = DecodingPress(
        base_press=KnormPress(compression_ratio=0.5),
        compression_steps=compression_steps,
        token_buffer_size=40
    )
    
    # Test context and question
    context = "The quick brown fox jumps over the lazy dog. " * 6
    question = "What animal jumps over the dog?"
    
    # Run pipeline
    cache = DynamicCache()
    result = pipe(context, question=question, press=press, cache=cache, max_new_tokens=15)
    
    # Check final cache size (should always be token_buffer_size regardless of compression_steps)
    for layer_idx, key_tensor in enumerate(cache.key_cache):
        layer_seq_len = key_tensor.shape[2]
        assert 40 <= layer_seq_len <= 40 + 1, (
            f"Layer {layer_idx}: Expected cache size to be 40, but got {layer_seq_len}"
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
