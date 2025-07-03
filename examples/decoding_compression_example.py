#!/usr/bin/env python3
"""
Example demonstrating the new compression during decoding interface.

This example shows how to use:
1. DecodingPress - for compression only during decoding
2. PrefillDecodingPress - for combining prefilling and decoding compression
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from kvpress.presses import (
    DecodingPress, 
    PrefillDecodingPress,
    ScorerPress,
    ExpectedAttentionPress,
    SnapKVPress
)
from kvpress.pipeline import KVPressTextGenerationPipeline


def main():
    # Load a small model for demonstration
    model_name = "microsoft/DialoGPT-small"  # Small model for quick testing
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create the pipeline
    pipeline = KVPressTextGenerationPipeline(
        model=model,
        tokenizer=tokenizer
    )
    
    # Example 1: DecodingPress only
    print("\n=== Example 1: DecodingPress only ===")
    
    # Create a scorer press for computing importance scores
    scorer_press = ScorerPress(
        scorer="expected_attention",
        compression_ratio=0.3,  # Remove 30% of tokens
    )
    
    # Create a decoding press that compresses every 5 steps
    decoding_press = DecodingPress(
        scorer_press=scorer_press,
        compression_steps=5,
        compression_ratio=0.3
    )
    
    context = "The weather today is sunny and warm. Many people are enjoying outdoor activities."
    question = "What is the weather like?"
    
    result = pipeline(
        context=context,
        question=question,
        press=decoding_press,
        max_new_tokens=20
    )
    
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Answer: {result['answers'][0]}")
    
    # Example 2: PrefillDecodingPress combining both phases
    print("\n=== Example 2: PrefillDecodingPress (combined) ===")
    
    # Create a prefilling press (e.g., SnapKV for prefilling)
    prefilling_press = SnapKVPress(
        compression_ratio=0.4,  # Remove 40% during prefilling
        window_size=32
    )
    
    # Create a different decoding press
    decoding_scorer = ScorerPress(
        scorer="expected_attention",
        compression_ratio=0.2,  # Remove 20% during decoding
    )
    
    decoding_press_2 = DecodingPress(
        scorer_press=decoding_scorer,
        compression_steps=3,  # Compress every 3 steps
        compression_ratio=0.2
    )
    
    # Combine them
    combined_press = PrefillDecodingPress(
        prefilling_press=prefilling_press,
        decoding_press=decoding_press_2
    )
    
    context_long = """
    Artificial intelligence has made remarkable progress in recent years. 
    Machine learning algorithms can now perform tasks that were once thought 
    to be exclusively human, such as image recognition, natural language 
    processing, and strategic game playing. Deep learning, in particular, 
    has revolutionized many fields by enabling computers to learn complex 
    patterns from large amounts of data.
    """
    
    question_long = "What has AI achieved recently?"
    
    result_combined = pipeline(
        context=context_long.strip(),
        question=question_long,
        press=combined_press,
        max_new_tokens=30
    )
    
    print(f"Context: {context_long.strip()}")
    print(f"Question: {question_long}")
    print(f"Answer: {result_combined['answers'][0]}")
    
    # Example 3: DecodingPress only (no prefilling compression)
    print("\n=== Example 3: DecodingPress only (no prefilling) ===")
    
    decoding_only_press = PrefillDecodingPress(
        prefilling_press=None,  # No prefilling compression
        decoding_press=decoding_press
    )
    
    result_decoding_only = pipeline(
        context=context_long.strip(),
        question="How has deep learning helped?",
        press=decoding_only_press,
        max_new_tokens=25
    )
    
    print(f"Question: How has deep learning helped?")
    print(f"Answer: {result_decoding_only['answers'][0]}")


if __name__ == "__main__":
    main()
