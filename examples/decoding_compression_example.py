#!/usr/bin/env python3
"""
Example demonstrating the new compression during decoding interface.

This example shows how to use:
1. DecodingPress - for compression only during decoding
2. PrefillDecodingPress - for combining prefilling and decoding compression

IMPORTANT: Only certain presses are compatible with DecodingPress.
See DecodingPress docstring for detailed compatibility information.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from kvpress.presses import (
    DecodingPress, 
    PrefillDecodingPress,
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
    
    # Example 1: DecodingPress with compatible scorer
    print("\n=== Example 1: DecodingPress with KnormPress (compatible) ===")
    
    # Create a compatible scorer press (KnormPress only uses key norms)
    scorer_press = KnormPress(
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
    
    try:
        result = pipeline(
            context=context,
            question=question,
            press=decoding_press,
            max_new_tokens=20
        )
        
        print(f"Context: {context}")
        print(f"Question: {question}")
        print(f"Answer: {result['answers'][0]}")
    except Exception as e:
        print(f"Error with KnormPress: {e}")
    
    # Example 2: DecodingPress with RandomPress (always compatible)
    print("\n=== Example 2: DecodingPress with RandomPress (always compatible) ===")
    
    random_scorer = RandomPress(compression_ratio=0.2)
    
    decoding_press_random = DecodingPress(
        scorer_press=random_scorer,
        compression_steps=3,
        compression_ratio=0.2
    )
    
    try:
        result_random = pipeline(
            context=context,
            question="How is the weather?",
            press=decoding_press_random,
            max_new_tokens=15
        )
        
        print(f"Question: How is the weather?")
        print(f"Answer: {result_random['answers'][0]}")
    except Exception as e:
        print(f"Error with RandomPress: {e}")
    
    # Example 3: PrefillDecodingPress combining compatible presses
    print("\n=== Example 3: PrefillDecodingPress (prefill + decoding) ===")
    
    # Use StreamingLLMPress for prefilling (compatible)
    prefilling_press = StreamingLLMPress(
        compression_ratio=0.4,  # Remove 40% during prefilling
        n_sink=4  # Keep 4 sink tokens
    )
    
    # Use KnormPress for decoding
    decoding_scorer = KnormPress(compression_ratio=0.2)
    decoding_press_2 = DecodingPress(
        scorer_press=decoding_scorer,
        compression_steps=3,
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
    
    try:
        result_combined = pipeline(
            context=context_long.strip(),
            question="What has AI achieved recently?",
            press=combined_press,
            max_new_tokens=30
        )
        
        print(f"Context: {context_long.strip()}")
        print(f"Question: What has AI achieved recently?")
        print(f"Answer: {result_combined['answers'][0]}")
    except Exception as e:
        print(f"Error with combined press: {e}")
    
    # Example 4: Demonstrate incompatible press (will show warning)
    print("\n=== Example 4: Warning about incompatible presses ===")
    print("WARNING: The following presses are NOT compatible with DecodingPress:")
    print("- AdaKVPress, CriticalAdaKVPress (attention masking conflicts)")
    print("- KeyRerotationPress, FinchPress (position embedding dependencies)")
    print("- SnapKVPress, TOVAPress, ThinKPress (window attention dependencies)")
    print("- DuoAttentionPress (streaming pattern conflicts)")
    print("- SimLayerKVPress (lazy evaluation conflicts)")
    print("\nUse only simple scorers like KnormPress, RandomPress, or StreamingLLMPress")
    print("with DecodingPress for reliable results.")


if __name__ == "__main__":
    main()
