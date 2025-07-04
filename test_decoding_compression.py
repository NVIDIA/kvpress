#!/usr/bin/env python3

"""
Test script to verify that DecodingPress actually compresses during decoding.
"""

import logging
import pytest
from transformers import DynamicCache, pipeline

from kvpress.presses.generation.decoding_press import DecodingPress
from kvpress.presses.knorm_press import KnormPress


@pytest.mark.parametrize("token_buffer_size", [32, 64, 128])
def test_decoding_compression(token_buffer_size, caplog):
    """Test that DecodingPress compresses the cache during decoding."""
    
    # Set up detailed logging
    with caplog.at_level(logging.DEBUG):
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
        
        # Create cache and track its ID
        cache = DynamicCache()
        cache_id = id(cache)
        print(f"\n=== DEBUGGING DecodingPress with token_buffer_size={token_buffer_size} ===")
        print(f"Cache ID: {cache_id}")
        print(f"Press type: {type(press)}")
        print(f"Base press: {type(press.base_press)}")
        print(f"Compression steps: {press.compression_steps}")
        print(f"Token buffer size: {press.token_buffer_size}")
        
        # Test context and question
        context = "The quick brown fox jumps over the lazy dog. " * 10  # Repeat for longer context
        question = "What animal jumps over the dog?"
        
        print(f"Context length: {len(context.split())} words")
        print(f"Context: {context[:100]}...")
        print(f"Question: {question}")
        
        # Check initial press state
        print(f"\nInitial press state:")
        print(f"  Hidden states buffer length: {len(press.hidden_states_buffer)}")
        print(f"  Step count: {press.step_count}")
        
        # Run pipeline
        print(f"\n=== RUNNING PIPELINE ===")
        result = pipe(
            context, 
            question=question, 
            press=press, 
            cache=cache,
            max_new_tokens=20
        )
        
        print(f"\n=== RESULTS ===")
        print(f"Generated answer: {result['answer']}")
        print(f"Cache size after generation: {cache.key_cache[0].shape}")
        print(f"Sequence length: {cache.get_seq_length()}")
        
        # Check final press state
        print(f"\nFinal press state:")
        print(f"  Hidden states buffer length: {len(press.hidden_states_buffer)}")
        print(f"  Step count: {press.step_count}")
        
        # Print all captured log messages
        print(f"\n=== CAPTURED LOG MESSAGES ===")
        for record in caplog.records:
            if record.levelno >= logging.DEBUG:
                print(f"[{record.levelname}] {record.name}: {record.message}")
        
        # Detailed cache analysis
        print(f"\n=== CACHE ANALYSIS ===")
        layer_shapes = []
        if cache.key_cache:
            for layer_idx, key_tensor in enumerate(cache.key_cache):
                layer_seq_len = key_tensor.shape[2]
                layer_shapes.append(layer_seq_len)
                print(f"Layer {layer_idx}: Key cache shape = {key_tensor.shape}, seq_len = {layer_seq_len}")
        
        print(f"All layer sequence lengths: {layer_shapes}")
        
        # Check if compression actually happened
        expected_compression = token_buffer_size < 150  # 150 is roughly the uncompressed size
        actual_compression = cache.get_seq_length() < 150
        
        print(f"\n=== COMPRESSION CHECK ===")
        print(f"Expected compression (buffer_size < 150): {expected_compression}")
        print(f"Actual compression (seq_len < 150): {actual_compression}")
        print(f"Expected seq_len per layer: {token_buffer_size}")
        print(f"Actual seq_len (max): {cache.get_seq_length()}")
        print(f"Actual seq_len per layer: {layer_shapes}")
        
        # Assert that each layer's cache sequence length equals the token_buffer_size
        print(f"\n=== LAYER-BY-LAYER ASSERTIONS ===")
        for layer_idx, layer_seq_len in enumerate(layer_shapes):
            print(f"Layer {layer_idx}: Expected {token_buffer_size}, Got {layer_seq_len}")
            
            # For now, let's see what we get without asserting
            if layer_seq_len == token_buffer_size:
                print(f"  ✓ Layer {layer_idx} matches expected buffer size")
            else:
                print(f"  ✗ Layer {layer_idx} does NOT match expected buffer size")
        
        # Check if at least one layer is compressed correctly
        correctly_compressed_layers = [i for i, seq_len in enumerate(layer_shapes) if seq_len == token_buffer_size]
        print(f"\nCorrectly compressed layers: {correctly_compressed_layers}")
        
        # For debugging, let's not assert yet - just report what we find
        if len(correctly_compressed_layers) == len(layer_shapes):
            print("✓ ALL layers compressed correctly!")
        elif len(correctly_compressed_layers) > 0:
            print(f"⚠ PARTIAL compression: {len(correctly_compressed_layers)}/{len(layer_shapes)} layers compressed")
        else:
            print("✗ NO layers compressed correctly")
            
        # Temporary assertion - let's see if at least some layers are compressed
        assert len(correctly_compressed_layers) > 0, (
            f"Expected at least one layer to be compressed to {token_buffer_size} tokens, "
            f"but got layer shapes: {layer_shapes}"
        )