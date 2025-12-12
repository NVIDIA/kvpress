# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from kvpress.presses.kvzip_press import KVzipPress

logger = logging.getLogger(__name__)


@dataclass
class KVSummaryPress(KVzipPress):
    """
    KVSummary identifies the importance of KV pairs by first generating a summary
    of the context and then computing cross-attention between summary tokens and
    all KV pairs.

    Unlike KVzip which uses chunked context reconstruction to score KV pairs,
    KVSummary:
    1. Performs initial prefilling of the context
    2. Generates a summary of the context (decoding phase)
    3. Computes cross-attention between summary tokens and all KV pairs
    4. Uses attention scores to determine which KV pairs to retain

    This approach provides a query-agnostic compression that preserves KV pairs
    most relevant to a semantic summary of the content.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    layerwise : bool, default=False
        Whether to enable uniform compression ratios across layers.
    n_sink : int, default=4
        Number of initial tokens to preserve as attention sinks.
    max_summary_tokens : int, default=256
        Maximum number of tokens to generate for the summary.
    summary_prompt : str, default=None
        Custom prompt to generate the summary. If None, uses a default prompt.
    """

    max_summary_tokens: int = 256
    summary_prompt: str | None = None
    
    # Internal flag to protect generation from the compression hook
    _is_generating: bool = field(default=False, repr=False)

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        self._reset_internal_parameters()
        self._is_generating = False

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        CRITICAL OVERRIDE:
        We must intercept the hook. If we are currently generating the summary,
        we must bypass the parent's hook completely.
        
        If we don't, the parent hook will forcibly truncate the KV cache 
        to `self.context_length` after every token, deleting the summary 
        as it is being created.
        """
        if self._is_generating:
            return output
            
        return super().forward_hook(module, input, kwargs, output)

    def prepare(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        chunk_size: int = 2048,
        prev_postfix_size: int = 8,
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """
        Override parent's prepare to generate summary instead of chunked reconstruction prompts.

        Returns a single "chunk" where the summary tokens are used for scoring
        the entire context (from prefix_length to context_length).
        """
        # Step 1: Initialize scores to 1.0 (Keep All) for safety
        self.score_val = torch.ones(
            (
                model.config.num_hidden_layers,
                1,
                model.config.num_key_value_heads,
                self.context_length,
            ),
            dtype=model.dtype,
            device=model.device,
        )

        # Step 2: Enable the protection flag
        # This tells forward_hook to do NOTHING while we generate
        self._is_generating = True
        
        summary_ids = None

        try:
            # Step 3: Generate the summary
            # The hook is now disabled, so the cache can grow naturally
            summary_ids, tokenizer_ref = self._generate_summary(model, tokenizer)
            
            summary_text = tokenizer_ref.decode(summary_ids[0], skip_special_tokens=True)
            print(f"[KVSummaryPress] Generated {summary_ids.shape[1]} tokens:\n{summary_text}", flush=True)

        finally:
            # Step 4: Disable protection flag
            # We are done generating. We need the hook active for the next phase (scoring).
            self._is_generating = False

        # Step 5: Trim cache back to original context length 
        # We manually remove the summary tokens from the cache now that we have the IDs
        for layer_idx in range(len(self._cache)):
            if hasattr(self._cache, "layers"):
                cache_layer = self._cache.layers[layer_idx]
                cache_layer.keys = cache_layer.keys[:, :, : self.context_length]
                cache_layer.values = cache_layer.values[:, :, : self.context_length]
            elif hasattr(self._cache, "key_cache"):  # HF DynamicCache
                self._cache.key_cache[layer_idx] = self._cache.key_cache[layer_idx][:, :, : self.context_length]
                self._cache.value_cache[layer_idx] = self._cache.value_cache[layer_idx][:, :, : self.context_length]
            elif isinstance(self._cache, list):
                self._cache[layer_idx] = (
                    self._cache[layer_idx][0][:, :, : self.context_length],
                    self._cache[layer_idx][1][:, :, : self.context_length]
                )

        # Step 6: Reset cache metadata (Position ID synchronization)
        if hasattr(self._cache, "_seen_tokens"):
            self._cache._seen_tokens = self.context_length
        elif hasattr(self._cache, "seen_tokens"):
            self._cache.seen_tokens = self.context_length

        # Step 7: Reset scores to ZERO for the actual scoring calculation
        self.score_val.zero_()
        self.score_val[..., : self.n_sink] = 1.0

        # Step 8: Return the summary tokens to be used as the "query"
        # The parent loop will now run a forward pass with these IDs.
        # The hook (now active) will compute attention between these IDs and the cache.
        ctx_len = self.context_length - self.prefix_length
        dummy_prefill_ids = torch.ones((1, ctx_len), dtype=torch.long, device=model.device)
        
        return [(dummy_prefill_ids, summary_ids)]

    def _generate_summary(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> tuple[torch.Tensor, PreTrainedTokenizer]:
        """
        Generate a summary using proper chat templates to avoid repetition loops.
        """
        prompt_text = self.summary_prompt or """
Task: Distill this document into a set of retrieval keys.

Extract specific entities and metrics as a single dense paragraph.
Rules:
1. Copy **UUIDs, Codes, and IDs** exactly.
2. Keep units with numbers (e.g., "100 years").
3. **DO NOT uses lists, bullet points, or newlines.**
4. Separate items with commas only.

Output format:
[Item 1], [Item 2], [Item 3], [Item 4]...
"""
        
        
        old_prompt_text = """
Task: Distill this document into a set of retrieval keys.

List every SINGLE specific fact, including:
   - All unique codes, IDs, or UUIDs (Copy exactly).
   - All numerical values and dates.
   - All proper names (People, Places, Orgs).
   - All distinct technical terms.

Output format:
[Item 1], [Item 2], [Item 3]...
        """
        # Use apply_chat_template to add proper <|start_header_id|>user and <|start_header_id|>assistant tokens
        # This is critical for instruction-tuned models to switch from completion to generation mode
        messages = [{"role": "user", "content": prompt_text}]
        
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,  # Critical: Adds the "Assistant:" start token
            return_tensors="pt"
        ).to(model.device)

        # Only strip BOS if we are appending to an existing sequence AND the tokenizer added one
        if prompt_ids[0, 0] == tokenizer.bos_token_id and self.context_length > 0:
            prompt_ids = prompt_ids[:, 1:]

        # Feed the prompt tokens in one batch (faster than token-by-token)
        current_position = self.context_length
        
        outputs = model(
            input_ids=prompt_ids,
            past_key_values=self._cache,
            position_ids=torch.arange(
                current_position, 
                current_position + prompt_ids.shape[1], 
                device=model.device
            ).unsqueeze(0),
            num_logits_to_keep=1,
        )
        current_position += prompt_ids.shape[1]

        # Greedy Decoding Loop
        generated_ids = [outputs.logits[0, -1].argmax()]
        
        should_stop_token_ids = model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids] if should_stop_token_ids is not None else []
            
        # Add EOT ID (End of Turn) if available, as chat models often use this instead of EOS
        if hasattr(tokenizer, "eot_token_id") and tokenizer.eot_token_id is not None:
            should_stop_token_ids.append(tokenizer.eot_token_id)
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            try:
                eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot_id is not None:
                    should_stop_token_ids.append(eot_id)
            except:
                pass

        for i in range(self.max_summary_tokens - 1):
            outputs = model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=self._cache,
                position_ids=torch.tensor([[current_position]], device=model.device),
            )
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)
            
            if new_id.item() in should_stop_token_ids:
                break
                
            current_position += 1

        return torch.stack(generated_ids).unsqueeze(0), tokenizer
