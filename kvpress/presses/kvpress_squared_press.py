# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field
from typing import List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from kvpress.presses.keydiff_press import KeyDiffPress
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class KVPressSquaredPress(KVzipPress):
    """
    KVPress^2 (KVPress Squared) combines a scorer press with KVzip reconstruction.

    Instead of doing reconstruction with all blocks of the document, it:
    1. Uses an inner press (default: KeyDiffPress with compression_ratio=0.8) to score tokens
    2. Selects the top (1 - inner_press.compression_ratio) = 20% highest-scoring positions
    3. Performs KVzip reconstruction using ONLY those selected tokens as queries
    4. But attends to ALL KV pairs in the original document during reconstruction
    5. Scores ALL KV pairs based on attention weights from the reconstruction

    This significantly reduces the computational overhead of KVzip (only 20% of tokens
    are reconstructed) while still scoring all KV pairs in the document.

    Based on KVzip (https://arxiv.org/abs/2505.23416).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression (applied by KVzip).
    layerwise : bool, default=False
        Whether to enable uniform compression ratios across layers.
    n_sink : int, default=4
        Number of initial tokens to preserve as attention sinks.
    inner_press : ScorerPress, default=KeyDiffPress(compression_ratio=0.8)
        The inner press used to score and select tokens for reconstruction.
        The inner_press.compression_ratio determines what fraction of tokens to SKIP:
        - compression_ratio=0.8 means select top 20% for reconstruction
        - compression_ratio=0.5 means select top 50% for reconstruction
    """

    inner_press: ScorerPress = field(default_factory=lambda: KeyDiffPress(compression_ratio=0.9))

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.inner_press, ScorerPress), "inner_press must be a ScorerPress subclass"

        top_ratio = 1 - self.inner_press.compression_ratio
        # Override the warning from KVzipPress with a more accurate one
        logger.warning(
            "KVPressSquaredPress uses an inner press to select top tokens before KVzip reconstruction. "
            f"Only the top {top_ratio * 100:.0f}% of tokens will be used for reconstruction, "
            "significantly reducing computational overhead compared to standard KVzip."
        )

    def _reset_internal_parameters(self):
        super()._reset_internal_parameters()
        self._selected_positions = None
        self._current_chunk_positions = None

    def _compute_inner_press_scores(self, model: PreTrainedModel) -> torch.Tensor:
        """
        Compute scores using the inner press for all tokens across all layers.

        Returns aggregated scores with shape (batch_size, seq_len) for selecting top positions.
        """
        all_scores = []

        for layer_idx, layer in enumerate(model.model.layers):
            module = layer.self_attn
            keys, values = extract_keys_and_values(self._cache, layer_idx)

            # Create dummy hidden_states (not used by most scorer presses like KeyDiffPress)
            bsz, num_kv_heads, seq_len, head_dim = keys.shape
            hidden_states = torch.zeros(bsz, seq_len, model.config.hidden_size, device=keys.device, dtype=keys.dtype)

            # Compute scores: shape (batch_size, num_kv_heads, seq_len)
            scores = self.inner_press.score(
                module=module, hidden_states=hidden_states, keys=keys, values=values, attentions=None, kwargs={}
            )
            all_scores.append(scores.mean(dim=1))  # Average across heads

        # Average across layers: (batch_size, seq_len)
        return torch.stack(all_scores, dim=0).mean(dim=0)

    def _select_top_positions(self, scores: torch.Tensor) -> torch.Tensor:
        """Select the top positions based on inner_press.compression_ratio, returning sorted absolute indices."""
        bsz, seq_len = scores.shape
        assert bsz == 1, "Only batch size of 1 is supported"

        # top_ratio = 1 - compression_ratio (e.g., compression_ratio=0.8 -> select top 20%)
        top_ratio = 1 - self.inner_press.compression_ratio
        context_len = seq_len - self.prefix_length
        n_selected = max(int(context_len * top_ratio), 1)

        # Select top positions from context portion only
        top_indices = scores[:, self.prefix_length :].topk(n_selected, dim=-1).indices
        absolute_indices = (top_indices + self.prefix_length).squeeze(0)

        return absolute_indices.sort().values

    def _chunk_fn(self, ctx_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        """
        Override parent's _chunk_fn to chunk selected positions instead of contiguous ids.

        Note: For KVPressSquaredPress, this is called with position indices, not token ids.
        """
        n_positions = len(ctx_ids) if ctx_ids.dim() == 1 else ctx_ids.shape[1]

        if n_positions <= chunk_size:
            return [ctx_ids]

        chunk_num = (n_positions - 1) // chunk_size + 1
        chunks = []
        for i in range(chunk_num):
            start, end = i * chunk_size, min((i + 1) * chunk_size, n_positions)
            chunk = ctx_ids[start:end] if ctx_ids.dim() == 1 else ctx_ids[:, start:end]
            if len(chunk) > 0:
                chunks.append(chunk)
        return chunks

    def prepare(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        chunk_size: int = 2048,
        prev_postfix_size: int = 8,
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare chunked inputs for selected positions only.

        Overrides parent to use selected positions instead of all context tokens.
        """
        # Initialize score values (reuse parent's structure)
        self.score_val = torch.zeros(
            (model.config.num_hidden_layers, 1, model.config.num_key_value_heads, self.context_length),
            dtype=model.dtype,
            device=model.device,
        )
        self.score_val[..., : self.n_sink] = 1.0

        # Get context token ids and convert positions to relative (both on CPU)
        ctx_ids = self._context_ids[:, self.prefix_length :].to("cpu")
        relative_positions = (self._selected_positions - self.prefix_length).to("cpu")

        # Chunk the positions
        chunked_positions = self._chunk_fn(relative_positions, chunk_size)
        chunked_context_pairs = []

        for i, chunk_pos in enumerate(chunked_positions):
            a_ids = ctx_ids[:, chunk_pos]

            if i == 0:
                prompt = "\n\nRepeat the selected context tokens exactly."
                q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
            else:
                prompt = "\n\nRepeat the next selected context tokens, starting with"
                q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
                postfix_ids = ctx_ids[:, chunked_positions[i - 1][-prev_postfix_size:]]
                q_ids = torch.cat([q_ids, postfix_ids], dim=1)

            # Store chunk positions for score_kvzip (third element is our extension)
            chunked_context_pairs.append((a_ids, torch.cat([q_ids, self._suffix_ids, a_ids], dim=1), chunk_pos))

        return chunked_context_pairs

    def _perform_kvzip_compression(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Perform KVPress^2 scoring: first select top positions, then run KVzip on them.
        """
        self.context_length = self._context_ids.shape[1]

        # Step 1: Compute inner press scores and select top positions
        inner_scores = self._compute_inner_press_scores(model)
        self._selected_positions = self._select_top_positions(inner_scores)

        top_ratio = 1 - self.inner_press.compression_ratio
        print(f"[KVPressSquaredPress] Selected {len(self._selected_positions)} positions " \
              f"out of {self.context_length - self.prefix_length} context tokens ({top_ratio * 100:.0f}%)", flush=True)

        # Step 2: Prepare and run reconstruction on selected positions only
        chunked_context_pairs = self.prepare(model, tokenizer)

        for prefill_ids, repeat_ids, chunk_positions in chunked_context_pairs:
            # Store positions for score_kvzip to use
            self._current_chunk_positions = chunk_positions + self.prefix_length
            self.start_idx = self._current_chunk_positions[0].item()
            self.end_idx = self._current_chunk_positions[-1].item() + 1

            model(input_ids=repeat_ids.to(model.device), past_key_values=self._cache, num_logits_to_keep=1)

        # Step 3: Final compression using parent's method
        self.compress_post(model)

    def _get_keys_for_scoring(self, keys: torch.Tensor, sink: int, q_len: int) -> torch.Tensor:
        """Get keys matching parent's structure: [sink] + [all context after sink] + [repeat]."""
        return torch.cat([
            keys[:, :, :sink],  # sink tokens
            keys[:, :, sink : self.context_length],  # all context tokens after sink
            keys[:, :, -q_len:],  # repeat chunk
        ], dim=2)

    def _get_ctx_len(self) -> int:
        """Get context length (after sink) - score ALL context positions."""
        sink = min(self.n_sink, self.start_idx)
        return self.context_length - sink

    def _assign_scores(self, layer_idx: int, sink: int, ctx_len: int, scores: torch.Tensor):
        """Assign scores to ALL context positions (after sink) using max aggregation."""
        # Use max to aggregate scores across multiple reconstruction passes
        current_scores = self.score_val[layer_idx][:, :, sink : self.context_length]
        self.score_val[layer_idx][:, :, sink : self.context_length] = torch.maximum(current_scores, scores)
