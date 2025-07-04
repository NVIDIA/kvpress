#!/usr/bin/env python3

"""
Test script to verify that DecodingPress actually compresses during decoding.
"""

import pytest
from transformers import DynamicCache, pipeline

from kvpress.presses.generation.decoding_press import DecodingPress
from kvpress.presses.knorm_press import KnormPress


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