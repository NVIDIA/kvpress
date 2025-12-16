# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.keydiff_press import KeyDiffPress
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.presses.random_press import RandomPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.expected_attention_press import ExpectedAttentionPress
from kvpress.presses.pyramidkv_press import PyramidKVPress
from kvpress.utils import extract_keys_and_values, get_prerope_query_states

logger = logging.getLogger(__name__)


@dataclass
class KVSquaredPress(KVzipPress):
    """
    KV² (KV-Squared): A two-stage KV cache compression framework extending KVzip.

    For each chunk (size 2048):
    Stage 1: Score tokens using inner_press (e.g., KeyDiffPress) and select top tokens.
    Stage 2: Use selected tokens as queries to attend to the SAME chunk's KV pairs,
             computing reconstruction scores (like KVzip).

    Based on KVzip (https://arxiv.org/abs/2505.23416).

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
    query_compression_ratio : float, default=0.5
        Per-chunk ratio for selecting reconstruction queries.
        - 0.5 means select top 50% of each chunk as queries
        - 0.9 means select top 10% of each chunk as queries
    """

    inner_press: ScorerPress = field(default_factory=lambda: KeyDiffPress())
    query_compression_ratio: float = 0.98

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.inner_press, ScorerPress), "inner_press must be a ScorerPress subclass"
        top_ratio = 1 - self.query_compression_ratio
        logger.warning(
            f"KV² initialized. Top {top_ratio * 100:.0f}% of each chunk selected for reconstruction."
        )

    def _reset_internal_parameters(self):
        super()._reset_internal_parameters()
        self._current_chunk_start = 0
        self._current_chunk_end = 0
        self._current_selected_positions = None

    def _compute_chunk_scores(self, model: PreTrainedModel, chunk_start: int, chunk_end: int) -> torch.Tensor:
        """Compute aggregated importance scores for a specific chunk across all layers/heads."""
        all_scores = []
        for layer_idx, layer in enumerate(model.model.layers):
            keys, values = extract_keys_and_values(self._cache, layer_idx)
            bsz, _, seq_len, _ = keys.shape

            # Extract only the chunk's keys
            chunk_keys = keys[:, :, chunk_start:chunk_end, :]
            chunk_values = values[:, :, chunk_start:chunk_end, :]
            chunk_len = chunk_end - chunk_start

            # Dummy hidden_states required by ScorerPress signature
            hidden_states = torch.zeros(
                bsz, chunk_len, model.config.hidden_size, device=keys.device, dtype=keys.dtype
            )

            scores = self.inner_press.score(
                module=layer.self_attn,
                hidden_states=hidden_states,
                keys=chunk_keys,
                values=chunk_values,
                attentions=None,
                kwargs={},
            )
            all_scores.append(scores.mean(dim=1))  # Mean over heads

        return torch.stack(all_scores).mean(dim=0)  # Mean over layers

    def _select_top_positions_in_chunk(self, scores: torch.Tensor, chunk_start: int) -> torch.Tensor:
        """Select top positions within a chunk based on scores."""
        chunk_len = scores.shape[-1]
        top_ratio = 1 - self.query_compression_ratio
        n_selected = max(int(chunk_len * top_ratio), 1)

        top_indices = scores.topk(n_selected, dim=-1).indices

        # Map back to absolute positions and sort
        return (top_indices + chunk_start).squeeze(0).sort().values

    def _perform_kvzip_compression(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """Execute per-chunk: Stage 1 (Score + Select) -> Stage 2 (Reconstruction) -> Compression."""
        self.context_length = self._context_ids.shape[1]

        # Move context_ids to GPU once (avoid repeated device transfers)
        self._context_ids = self._context_ids.to(model.device)

        # Use parent's prepare to get chunk boundaries
        chunked_context_pairs = super().prepare(model, tokenizer)

        # Process each chunk
        self.start_idx = self.prefix_length
        for prefill_ids, repeat_ids in chunked_context_pairs:
            chunk_start = self.start_idx
            chunk_end = chunk_start + prefill_ids.shape[1]
            self.end_idx = chunk_end

            # Stage 1: Compute scores and select top tokens for this chunk
            chunk_scores = self._compute_chunk_scores(model, chunk_start, chunk_end)
            selected_positions = self._select_top_positions_in_chunk(chunk_scores, chunk_start)

            # Store for use in score_kvzip
            self._current_chunk_start = chunk_start
            self._current_chunk_end = chunk_end
            self._current_selected_positions = selected_positions

            # Stage 2: Pass only selected tokens (no prompt/suffix needed)
            # HF assigns contiguous positions [past_len, past_len+1, ..., past_len+q_len-1]
            # which matches our keys[:, :, -q_len:] in score_kvzip
            pos = selected_positions.to(self._context_ids.device)
            selected_ids = self._context_ids.index_select(1, pos)  # (bsz, q_len)

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

    def score_kvzip(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention scores for KV pairs using selected reconstruction queries.
        Like KVzip: attends only to the current chunk's keys (not entire context).
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads, head_dim = module.config.num_attention_heads, module.head_dim
        num_kv_heads = module.config.num_key_value_heads

        # Prepare Queries
        queries = get_prerope_query_states(module, hidden_states)
        cos, sin = kwargs["position_embeddings"]
        queries = (queries * cos.unsqueeze(1)) + (rotate_half(queries) * sin.unsqueeze(1))
        queries = queries.view(bsz, num_kv_heads, num_heads // num_kv_heads, q_len, head_dim)

        # Prepare Keys: [Sink] + [Chunk Keys] + [Repeat Queries]
        # Like KVzip: attend only to current chunk's keys
        sink = min(self.n_sink, self.start_idx)
        ctx_len = self.end_idx - self.start_idx
        keys_subsampled = torch.cat(
            [
                keys[:, :, :sink],
                keys[:, :, self.start_idx : self.end_idx],  # Current chunk only
                keys[:, :, -q_len:],
            ],
            dim=2,
        ).unsqueeze(2).transpose(-2, -1).contiguous()

        # Attention
        attn_weights = torch.matmul(queries, keys_subsampled) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Extract scores for chunk (after sink, before repeat)
        scores = attn_weights[..., sink : sink + ctx_len].amax(dim=(-3, -2))

        # Update scores for this chunk (direct assignment like KVzip)
        layer_idx = int(module.layer_idx)
        self.score_val[layer_idx][..., self.start_idx : self.end_idx] = scores

        # Return original keys/values (stripping the repeat chunk)
        return keys[:, :, : self.context_length], values[:, :, : self.context_length]
