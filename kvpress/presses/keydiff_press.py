# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer

from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


@dataclass
class KeyDiffPress(ScorerPress):
    """
    KeyDiff: Key similarity-based KV cache compression.

    Evicts tokens based on key vector similarity to average key pattern.
    Identifies tokens with most similar keys to average and removes them,
    keeping tokens with more distinctive key vectors.

    Based on KeyDiff (https://arxiv.org/abs/2504.15364).

    Note: The original press in the KeyDiff paper implements a block-wise iterative compression.
    In KVPress, the iterative compression is implemented in the BlockPress class.
    Therefore, to replicate the paper's implementation, please use:

    `press = BlockPress(press=KeyDiffPress(compression_ratio=0.x), block_size=N)`

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    measure_decoding_latency : bool, default=False
        If True, runs decoding forward passes after prefilling to measure latency.
    num_decoding_tokens : int, default=100
        Number of decoding tokens to generate for latency measurement.
    decoding_prompt : str, optional
        Prompt to use for decoding (will be tokenized and used as starting tokens).
        If None, uses dummy tokens for pure latency measurement.
    """

    measure_decoding_latency: bool = False
    num_decoding_tokens: int = 100
    decoding_prompt: Optional[str] = None
    
    # Internal state (not user-configurable)
    _model: PreTrainedModel = field(default=None, repr=False)
    _tokenizer: PreTrainedTokenizer = field(default=None, repr=False)
    _decoding_latency_measured: bool = field(default=False, repr=False)
    _total_decoding_time: float = field(default=0.0, repr=False)
    _num_layers: int = field(default=0, repr=False)
    _layers_processed: int = field(default=0, repr=False)
    _prefill_start_time: float = field(default=0.0, repr=False)
    _sample_count: int = field(default=0, repr=False)
    _current_sample_id: Optional[str] = field(default=None, repr=False)

    def post_init_from_model(self, model: PreTrainedModel):
        """Store reference to model for decoding latency measurement."""
        self._model = model
        self._decoding_latency_measured = False
        self._total_decoding_time = 0.0
        self._prefill_start_time = 0.0
        # Don't reset _sample_count here - let it persist across samples
        # Get number of layers from the model
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        self._num_layers = len(language_model.layers)
        self._layers_processed = 0
    
    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Set the tokenizer for decoding prompt and generated text."""
        self._tokenizer = tokenizer
    
    def set_current_sample_id(self, sample_id: str):
        """Set the current sample ID for logging purposes."""
        self._current_sample_id = sample_id

    def _clone_cache(self, cache) -> DynamicCache:
        """Create a deep copy of the cache for latency measurement."""
        cloned_cache = DynamicCache()
        for layer in cache.layers:
            # Clone the key and value tensors
            cloned_cache.update(
                layer.keys.clone(),
                layer.values.clone(),
                layer_idx=len(cloned_cache.layers),
            )
        return cloned_cache

    def _get_formatted_prompt(self, prompt: str) -> str:
        """
        Format the prompt using the model's chat template.
        
        This extracts the suffix from the chat template (the part that signals
        the assistant's turn) and appends it to the prompt.
        """
        if self._tokenizer is None:
            return prompt
        
        if self._tokenizer.chat_template is None:
            # No chat template, use simple newline suffix
            return prompt + "\n"
        
        # Use a dummy context to extract the question suffix from chat template
        dummy_context = "dummy context"
        separator = "\n" + "#" * len(dummy_context)
        temp_context = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": dummy_context + separator}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        _, suffix_text = temp_context.split(separator)
        
        # Return prompt + suffix (suffix signals assistant's turn)
        return prompt + suffix_text

    def _measure_single_generation(self, cache, cache_length: int, prompt_ids, device) -> tuple[float, list[int]]:
        """Measure decoding latency for generation. Returns (time, generated_ids)."""
        
        # Warm-up run (not timed)
        warmup_token_id = prompt_ids[0, 0].item() if prompt_ids is not None else 1
        warmup_input = torch.tensor([[warmup_token_id]], device=device)
        warmup_position = torch.tensor([[cache_length]], device=device)
        with torch.no_grad():
            self._model(
                input_ids=warmup_input,
                past_key_values=cache,
                position_ids=warmup_position,
                use_cache=True,
            )
        
        # Synchronize before timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        generated_ids = []
        current_position = cache_length + 1  # After warmup token
        
        with torch.no_grad():
            # First, process the rest of the prompt if provided
            prompt_offset = 0
            if prompt_ids is not None and prompt_ids.shape[1] > 1:
                for i in range(1, prompt_ids.shape[1]):
                    input_ids = prompt_ids[:, i:i+1]
                    position_ids = torch.tensor([[current_position]], device=device)
                    
                    outputs = self._model(
                        input_ids=input_ids,
                        past_key_values=cache,
                        position_ids=position_ids,
                        use_cache=True,
                    )
                    current_position += 1
                    prompt_offset += 1
            
            # Get the first token to generate from the last prompt token
            if prompt_ids is not None:
                next_token = outputs.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            else:
                next_token = torch.tensor([[1]], device=device)
            
            # Generate remaining tokens
            tokens_to_generate = self.num_decoding_tokens - prompt_offset
            for i in range(tokens_to_generate):
                position_ids = torch.tensor([[current_position]], device=device)
                
                outputs = self._model(
                    input_ids=next_token,
                    past_key_values=cache,
                    position_ids=position_ids,
                    use_cache=True,
                )
                
                next_token_id = outputs.logits[0, -1].argmax()
                generated_ids.append(next_token_id.item())
                next_token = next_token_id.unsqueeze(0).unsqueeze(0)
                current_position += 1
                
                # Check for EOS token
                if self._tokenizer is not None:
                    eos_token_ids = self._model.generation_config.eos_token_id
                    if not isinstance(eos_token_ids, list):
                        eos_token_ids = [eos_token_ids]
                    if next_token_id.item() in eos_token_ids:
                        break
        
        # Synchronize after timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        elapsed_time = time.perf_counter() - start_time
        return elapsed_time, generated_ids

    def _measure_decoding_latency(self, cache, context_length: int):
        """
        Run forward passes to measure decoding latency and generate tokens.
        
        Parameters
        ----------
        cache : Cache
            The KV cache after prefilling (will be cloned, not modified).
        context_length : int
            The length of the context (sequence length after prefilling).
        """
        if self._model is None:
            logger.warning("Model reference not set, skipping decoding latency measurement")
            return
        
        device = next(self._model.parameters()).device
        
        # Clone the cache so we don't modify the original
        cloned_cache = self._clone_cache(cache)
        
        # Tokenize the decoding prompt with proper chat formatting if provided
        prompt_ids = None
        if self.decoding_prompt and self._tokenizer is not None:
            formatted_prompt = self._get_formatted_prompt(self.decoding_prompt)
            prompt_ids = self._tokenizer.encode(
                formatted_prompt, 
                return_tensors="pt", 
                add_special_tokens=False
            ).to(device)
        
        # Increment sample count
        self._sample_count += 1
        
        # Format sample identifier
        sample_id_str = f" (ID: {self._current_sample_id})" if self._current_sample_id else ""
        
        # Measure latency
        decoding_time, generated_ids = self._measure_single_generation(
            cloned_cache, context_length, prompt_ids, device
        )
        
        self._total_decoding_time = decoding_time
        
        # Print results
        total_tokens = len(generated_ids)
        avg_time_per_token = decoding_time / max(total_tokens, 1) * 1000
        
        if self._tokenizer is not None and generated_ids:
            generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"\n[KeyDiffPress] Sample {self._sample_count}{sample_id_str} - Generated text:")
            print(f"  Prompt: \"{self.decoding_prompt or '(none)'}\"")
            print(f"  Output: \"{generated_text}\"")
            print(f"  Tokens generated: {total_tokens}")
        
        print(
            f"[KeyDiffPress] Decoding (KV cache={context_length} tokens): {total_tokens} tokens in "
            f"{decoding_time:.4f}s ({avg_time_per_token:.2f} ms/token)"
        )
        
        # Clean up
        del cloned_cache

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Forward hook that optionally measures decoding latency after compression.
        
        If `measure_decoding_latency` is True, this hook will first compress the KV cache,
        then run forward passes to generate `num_decoding_tokens` tokens.
        This measures decoding latency with the COMPRESSED KV cache.
        """
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        
        # Don't compress after pre-filling
        if kwargs["cache_position"][-1] > q_len:
            return output
        
        # Track how many layers have been processed
        self._layers_processed += 1
        
        # Record prefill start time on first layer
        if self.measure_decoding_latency and self._layers_processed == 1:
            # Reset the flag for each new sample (layer count resets per sample)
            self._decoding_latency_measured = False
            device = next(self._model.parameters()).device
            if device.type == "cuda":
                torch.cuda.synchronize()
            self._prefill_start_time = time.perf_counter()
        
        # First, call parent's forward_hook to do the actual compression
        result = super().forward_hook(module, input, kwargs, output)
        
        # Measure decoding latency after the LAST layer has been COMPRESSED
        # This ensures we measure on the compressed cache
        if (self.measure_decoding_latency and 
            self._layers_processed == self._num_layers and 
            not self._decoding_latency_measured):
            
            # Calculate prefill time (includes compression time)
            device = next(self._model.parameters()).device
            if device.type == "cuda":
                torch.cuda.synchronize()
            prefill_time = time.perf_counter() - self._prefill_start_time
            
            # Get the compressed cache length
            compressed_length = cache.get_seq_length()
            
            # Print prefill stats
            print(
                f"\n[KeyDiffPress] Prefill + Compression: {compressed_length} tokens in {prefill_time:.4f}s "
                f"({prefill_time / max(compressed_length, 1) * 1000:.2f} ms/token)"
            )
            
            self._measure_decoding_latency(cache, compressed_length)
            self._decoding_latency_measured = True
            
            # Reset layer count for next sample
            self._layers_processed = 0
        
        return result

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        anchor = F.normalize(keys, p=2, dim=-1).mean(dim=2, keepdim=True)
        return -F.cosine_similarity(keys, anchor, dim=-1)
