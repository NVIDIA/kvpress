# Generation Presses (Experimental)

## Overview

This folder provides presses that can be used to compress the KV cache during token generation.
Currently, it supports a `DecodingPress` press that compresses the KV cache periodically during token generation.
Additionally, it provides a `PrefillDecodingPress` press that combines separate presses for prefilling and decoding phases.
This is an experimental feature and may change in future versions.

## Available Presses

### DecodingPress

Compresses the KV cache periodically during token generation by maintaining a buffer of recent hidden states.

**Key Features:**
- Only operates during decoding (not prefilling)
- Buffers hidden states from recent decoding steps
- Applies compression every N steps using any ScorerPress (note that some ScorerPresses may be incompatible with this press)
- Maintains target cache size throughout generation

**How it works:**
When compression is triggered, the scorer press receives:
- `keys`: Current key cache `[batch_size, n_heads, seq_len, head_dim]` (potentially already compressed from previous steps)
- `values`: Current value cache `[batch_size, n_heads, seq_len, head_dim]` (potentially already compressed from previous steps)
- `hidden_states`: Buffered hidden states `[batch_size, compression_steps, hidden_dim]` from recent decoding steps
- `attentions`: Attention weights (from current forward pass, may be None if `output_attentions` is not set explicitly)

Note that `hidden_states` contains only the recent `compression_steps` tokens, while `keys`/`values` contain the full sequence history (potentially already compressed from previous steps).

**Press Compatibility:**

Not all existing presses are fully compatible with DecodingPress due to fundamental differences in how compression works during decoding versus prefilling. 
In particular, DecodingPress provides buffered `hidden_states` containing only recent tokens (equal to `compression_steps`), while the `keys` and `values` contain the full sequence history.
Some presses may work reasonably well with this setup. For example, presses like `KNormPress` or `QFilterPress` that compute scores based solely on the `keys` or `values` tensors don't rely on sequence length alignment between `hidden_states` and the cache tensors.
Other presses can be used but may not provide optimal results. Presses such as `ExpectedAttentionPress` were specifically designed for prefilling scenarios where `hidden_states` represents the full input sequence. 
They may still be used for decoding. Future work may include revisiting these presses and adapt them specifically for the decoding phase.

**Parameters:**
- `base_press`: Any ScorerPress (e.g., `KNormPress`, `CriticalKVPress`)
- `compression_steps`: Steps between compressions (default: 10)
- `token_buffer_size`: Target cache size after compression (default: 1024)

### PrefillDecodingPress

Combines separate presses for prefilling and decoding phases.

**Parameters:**
- `prefilling_press`: Press used during prefill phase
- `decoding_press`: Press used during decoding phase

## Usage Examples

### Basic Decoding Compression

```python
from kvpress import KVPressTextGenerationPipeline
from kvpress.presses import KNormPress
from kvpress.presses.generation import DecodingPress

# Create a decoding press that compresses every 10 steps to 512 tokens
decoding_press = DecodingPress(
    base_press=KNormPress(compression_ratio=0.5),
    compression_steps=10,
    token_buffer_size=512
)

# Use with pipeline
pipeline = KVPressTextGenerationPipeline(model, tokenizer, press=decoding_press)
response = pipeline.generate_answer("Tell me a long story about...")
```

### Combined Prefill + Decoding Compression

```python
from kvpress.presses import CriticalKVPress
from kvpress.presses.generation import DecodingPress, PrefillDecodingPress

# Different strategies for prefill vs decoding
prefill_press = CriticalKVPress(compression_ratio=0.3)
decoding_press = DecodingPress(
    base_press=KNormPress(compression_ratio=0.4),
    compression_steps=5,
    token_buffer_size=256
)

# Combine them
combined_press = PrefillDecodingPress(
    prefilling_press=prefill_press,
    decoding_press=decoding_press
)

pipeline = KVPressTextGenerationPipeline(model, tokenizer, press=combined_press)
```

### Memory-Efficient Long Generation

```python
# For very long generation with tight memory constraints
decoding_press = DecodingPress(
    base_press=CriticalKVPress(compression_ratio=0.6),
    compression_steps=3,  # Compress more frequently
    token_buffer_size=128  # Keep cache very small
)

pipeline = KVPressTextGenerationPipeline(model, tokenizer, press=decoding_press)
long_response = pipeline.generate_answer(
    "Write a detailed analysis...",
    max_new_tokens=2000
)
```

## How It Works

1. **Prefill Phase**: Standard KV cache compression (if using PrefillDecodingPress)
2. **Decoding Phase**: 
   - Buffer hidden states from recent tokens
   - Every N steps, compress the entire cache using buffered states for context
   - Maintain target cache size throughout generation
   - Continue generating with compressed cache

## Performance Notes

- Compression frequency vs. quality tradeoff: more frequent compression (lower `compression_steps`) uses more compute but maintains smaller cache
- The `token_buffer_size` should be chosen based on available memory and desired context retention
- Some scorer presses could be optimized for decoding by caching intermediate scores

## Limitations

- **Experimental**: API may change in future versions
- Currently supports scorer-based presses only
- Compression overhead increases with frequency
- May affect generation quality depending on compression aggressiveness
