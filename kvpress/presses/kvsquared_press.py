# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from kvpress.presses.keydiff_press import KeyDiffPress
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class KVSquaredPress(KVzipPress):
    """
    KV² (KV-Squared): A two-stage KV cache compression framework.

    For each chunk:
        Stage 1: Score tokens using inner_press and select top tokens based on top_ratio.
        Stage 2: Run reconstruction pass with selected tokens to compute importance scores.

    Inherits infrastructure from KVzipPress (https://arxiv.org/abs/2505.23416).

    Self-Refining Mode
    ------------------
    Supports nesting for iterative refinement - each level adds a reconstruction pass:

        KVSquaredPress(inner_press=KVSquaredPress(inner_press=KeyDiffPress()))

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of KV pairs to remove during final compression.
    chunk_size : int, default=4096
        Size of chunks for processing context.
    inner_press : ScorerPress or KVSquaredPress, default=KeyDiffPress()
        Scoring mechanism for Stage 1. Any press with .score() or .compute_chunk_scores().
    top_ratio : float, default=0.02
        Per-chunk fraction of tokens selected as reconstruction queries.
    """

    chunk_size: int = 16384
    inner_press: ScorerPress | KVSquaredPress = field(default_factory=lambda: KeyDiffPress())
    top_ratio: float = 0.02

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be in [0, 1)"
        assert hasattr(self.inner_press, 'score') or hasattr(self.inner_press, 'compute_chunk_scores'), \
            "inner_press must have .score() or .compute_chunk_scores() method"
        assert 0 < self.top_ratio <= 1, "top_ratio must be in (0, 1]"
        self._reset_internal_parameters()

    def _get_language_model(self, model: PreTrainedModel):
        """Get the underlying language model, handling VLM wrappers."""
        return model.model.language_model if hasattr(model.model, "language_model") else model.model

    def _compute_chunk_scores(self, model: PreTrainedModel, chunk_start: int, chunk_end: int) -> torch.Tensor:
        """
        Stage 1: Score chunk tokens using inner_press.

        Supports duck-typing: inner_press can have either compute_chunk_scores()
        (for nested KVSquaredPress) or score() (for simple presses like KeyDiffPress).
        """
        if hasattr(self.inner_press, 'compute_chunk_scores'):
            return self.inner_press.compute_chunk_scores(
                model, self._cache, self._context_ids, chunk_start, chunk_end,
                self.prefix_length, self.context_length
            )

        language_model = self._get_language_model(model)
        all_scores = []
        for layer_idx, layer in enumerate(language_model.layers):
            keys, values = extract_keys_and_values(self._cache, layer_idx)
            scores = self.inner_press.score(
                module=layer.self_attn, hidden_states=None,
                keys=keys[:, :, chunk_start:chunk_end, :],
                values=values[:, :, chunk_start:chunk_end, :],
                attentions=None, kwargs={},
            )
            all_scores.append(scores.mean(dim=1))
        return torch.stack(all_scores).mean(dim=0)

    def _select_query_positions(self, scores: torch.Tensor, chunk_start: int) -> torch.Tensor:
        """Select top-scoring positions as reconstruction queries."""
        n_selected = max(int(scores.shape[-1] * self.top_ratio), 1)
        return (scores.topk(n_selected, dim=-1).indices + chunk_start).squeeze(0).sort().values

    def _run_chunk_reconstruction(self, model: PreTrainedModel, chunk_start: int, chunk_end: int):
        """
        Process a single chunk through both stages.

        Stage 1: Score tokens with inner_press, select top_ratio as queries
        Stage 2: Run reconstruction pass to compute importance via forward_hook
        """
        self.start_idx, self.end_idx = chunk_start, chunk_end

        # Stage 1: Score and select query positions
        scores = self._compute_chunk_scores(model, chunk_start, chunk_end)
        positions = self._select_query_positions(scores, chunk_start)
        selected_ids = self._context_ids.index_select(1, positions.to(self._context_ids.device))
        logger.debug(f"[KV²] Chunk [{chunk_start}:{chunk_end}] -> {len(positions)} queries")

        # Stage 2: Reconstruction pass (triggers forward_hook for importance scoring)
        with torch.inference_mode():
            model(input_ids=selected_ids, past_key_values=self._cache, use_cache=True, num_logits_to_keep=1)

    def _with_scoring_hooks(self, model: PreTrainedModel, fn):
        """Execute fn with forward hooks registered for importance scoring."""
        language_model = self._get_language_model(model)
        hooks = [layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True)
                 for layer in language_model.layers]
        try:
            return fn()
        finally:
            for h in hooks:
                h.remove()

    def compute_chunk_scores(self, model, cache, context_ids, chunk_start, chunk_end, prefix_length, context_length):
        """
        Compute chunk scores by running this press's full pipeline.
        Called when this KVSquaredPress is nested as inner_press of another.
        """
        self._cache, self._context_ids = cache, context_ids
        self.context_length, self.prefix_length = context_length, prefix_length
        self._init_score_val(model)

        self._with_scoring_hooks(model, lambda: self._run_chunk_reconstruction(model, chunk_start, chunk_end))

        return self.score_val[..., chunk_start:chunk_end].mean(dim=(0, 2))

    def _perform_kvzip_compression(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """Override parent to use KV² two-stage algorithm instead of text-based reconstruction."""
        self.context_length = self._context_ids.shape[1]
        self._context_ids = self._context_ids.to(model.device)
        self._init_score_val(model)

        # Process all chunks through Stage 1 + Stage 2
        pos = self.prefix_length
        for chunk in self._chunk_fn(self._context_ids[:, self.prefix_length:].cpu(), chunk_size=self.chunk_size):
            self._run_chunk_reconstruction(model, pos, pos + chunk.shape[1])
            pos += chunk.shape[1]

        self.compress_post(model)
