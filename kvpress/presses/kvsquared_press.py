# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from kvpress.presses.keydiff_press import KeyDiffPress
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.random_press import RandomPress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class KVSquaredPress(KVzipPress):
    """
    KV² (KV-Squared): A two-stage KV cache compression framework.

    For each chunk (size 2048):
    Stage 1: Score tokens using inner_press (e.g., KeyDiffPress) and select top tokens.
    Stage 2: Use selected tokens as queries to attend to the SAME chunk's KV pairs,
             computing reconstruction-based importance scores.

    Inherits infrastructure from KVzipPress (https://arxiv.org/abs/2505.23416).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during final compression.
    layerwise : bool, default=False
        Whether to enable uniform compression ratios across layers.
    n_sink : int, default=4
        Number of initial tokens to preserve as attention sinks.
    inner_press : ScorerPress, default=KeyDiffPress()
        The scoring mechanism used in Stage 1 to score tokens per chunk.
        Must be a scorer that doesn't require attention weights (e.g., KeyDiffPress, KnormPress).
        Note: Only the .score() method is used, so inner_press.compression_ratio is ignored.
    top_ratio : float, default=0.02
        Per-chunk fraction of tokens selected as reconstruction queries.
        - 0.50 means select top 50% of each chunk as queries
        - 0.10 means select top 10% of each chunk as queries
    """

    inner_press: ScorerPress = field(default_factory=lambda: KeyDiffPress())
    top_ratio: float = 0.05

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        assert isinstance(self.inner_press, ScorerPress), "inner_press must be a ScorerPress subclass"
        assert 0 < self.top_ratio <= 1, "top_ratio must be in (0, 1]"
        self._reset_internal_parameters()

    def _reset_internal_parameters(self):
        super()._reset_internal_parameters()

    def _compute_chunk_scores(self, model: PreTrainedModel, chunk_start: int, chunk_end: int) -> torch.Tensor:
        """Compute aggregated importance scores for a specific chunk across all layers/heads."""
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        all_scores = []
        for layer_idx, layer in enumerate(language_model.layers):
            keys, values = extract_keys_and_values(self._cache, layer_idx)
            bsz, _, _, _ = keys.shape

            # Extract only the chunk's keys
            chunk_keys = keys[:, :, chunk_start:chunk_end, :]
            chunk_values = values[:, :, chunk_start:chunk_end, :]

            try:
                scores = self.inner_press.score(
                    module=layer.self_attn,
                    hidden_states=None,  # KV² Stage-1 only supports KV-only scorers
                    keys=chunk_keys,
                    values=chunk_values,
                    attentions=None,
                    kwargs={},
                )
            except Exception as e:  # pragma: no cover (defensive: depends on user-provided inner_press)
                raise TypeError(
                    "KVSquaredPress(inner_press=...) only supports KV-only ScorerPress implementations "
                    "that do not require hidden_states/attentions/position_embeddings in .score()."
                ) from e
            all_scores.append(scores.mean(dim=1))  # Mean over heads

        return torch.stack(all_scores).mean(dim=0)  # Mean over layers

    def _select_top_positions_in_chunk(self, scores: torch.Tensor, chunk_start: int) -> torch.Tensor:
        """Select top positions within a chunk based on scores."""
        chunk_len = scores.shape[-1]
        n_selected = max(int(chunk_len * self.top_ratio), 1)
        top_indices = scores.topk(n_selected, dim=-1).indices

        # Map back to absolute positions and sort
        return (top_indices + chunk_start).squeeze(0).sort().values

    def _perform_kvzip_compression(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """Execute per-chunk: Stage 1 (Score + Select) -> Stage 2 (Reconstruction) -> Compression."""
        self.context_length = self._context_ids.shape[1]
        self._context_ids = self._context_ids.to(model.device)
        self._init_score_val(model)
        ctx_ids = self._context_ids[:, self.prefix_length :].to("cpu")
        chunked_input_ids = self._chunk_fn(ctx_ids, chunk_size=4096)

        # Process each chunk
        self.start_idx = self.prefix_length
        for prefill_ids in chunked_input_ids:
            chunk_start = self.start_idx
            chunk_end = chunk_start + prefill_ids.shape[1]
            self.end_idx = chunk_end

            # Stage 1: Compute scores and select top tokens for this chunk
            chunk_scores = self._compute_chunk_scores(model, chunk_start, chunk_end)
            selected_positions = self._select_top_positions_in_chunk(chunk_scores, chunk_start)

            # Stage 2: Pass only selected tokens (no prompt/suffix needed)
            pos = selected_positions.to(self._context_ids.device)
            selected_ids = self._context_ids.index_select(1, pos)

            logger.debug(f"[KV²] Chunk [{chunk_start}:{chunk_end}] -> {len(selected_positions)} queries")

            # Stage 2: Reconstruction pass with selected tokens only (contiguous positions)
            with torch.inference_mode():
                model(
                    input_ids=selected_ids,
                    past_key_values=self._cache,
                    use_cache=True,
                    num_logits_to_keep=1,
                )

            self.start_idx = chunk_end

        self.compress_post(model)
