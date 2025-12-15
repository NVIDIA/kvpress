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
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values, get_prerope_query_states

logger = logging.getLogger(__name__)


@dataclass
class KVSquaredPress(KVzipPress):
    """
    KV² (KV-Squared): A two-stage KV cache compression framework extending KVzip.

    Stage 1: Identify high-importance tokens using a lightweight scorer (e.g., KeyDiff).
    Stage 2: Perform KVzip reconstruction only using those selected tokens as queries,
             while attending to ALL KV pairs in the original context.

    Based on KVzip (https://arxiv.org/abs/2505.23416).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during final compression.
    layerwise : bool, default=False
        Whether to enable uniform compression ratios across layers.
    n_sink : int, default=4
        Number of initial tokens to preserve as attention sinks.
    inner_press : ScorerPress, default=KeyDiffPress(compression_ratio=0.5)
        The scoring mechanism used in Stage 1 to select tokens for reconstruction.
        The inner_press.compression_ratio determines what fraction of tokens to SKIP:
        - compression_ratio=0.9 means select top 10% for reconstruction
        - compression_ratio=0.5 means select top 50% for reconstruction
    """

    inner_press: ScorerPress = field(default_factory=lambda: KeyDiffPress(compression_ratio=0.8))

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.inner_press, ScorerPress), "inner_press must be a ScorerPress subclass"
        top_ratio = 1 - self.inner_press.compression_ratio
        logger.warning(
            f"KV² initialized. Top {top_ratio * 100:.0f}% of tokens selected for reconstruction."
        )

    def _reset_internal_parameters(self):
        super()._reset_internal_parameters()
        self._selected_positions = None
        self._current_chunk_positions = None

    def _compute_inner_press_scores(self, model: PreTrainedModel) -> torch.Tensor:
        """Stage 1: Compute aggregated importance scores across all layers/heads."""
        all_scores = []
        for layer_idx, layer in enumerate(model.model.layers):
            keys, values = extract_keys_and_values(self._cache, layer_idx)
            bsz, _, seq_len, _ = keys.shape

            # Dummy hidden_states required by ScorerPress signature
            hidden_states = torch.zeros(
                bsz, seq_len, model.config.hidden_size, device=keys.device, dtype=keys.dtype
            )

            scores = self.inner_press.score(
                module=layer.self_attn,
                hidden_states=hidden_states,
                keys=keys,
                values=values,
                attentions=None,
                kwargs={},
            )
            all_scores.append(scores.mean(dim=1))  # Mean over heads

        return torch.stack(all_scores).mean(dim=0)  # Mean over layers

    def _select_top_positions(self, scores: torch.Tensor) -> torch.Tensor:
        """Select indices of top-ranked tokens based on Stage 1 scores."""
        seq_len = scores.shape[-1]
        top_ratio = 1 - self.inner_press.compression_ratio
        n_selected = max(int((seq_len - self.prefix_length) * top_ratio), 1)

        # Score only the context part (skip prefix)
        context_scores = scores[:, self.prefix_length :]
        top_indices = context_scores.topk(n_selected, dim=-1).indices

        # Map back to absolute positions and sort
        return (top_indices + self.prefix_length).squeeze(0).sort().values

    def prepare(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        chunk_size: int = 2048,
        prev_postfix_size: int = 8,
    ) -> List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Overrides parent prepare to chunk only selected tokens.
        Temporarily mocks self._context_ids to reuse parent's prompt generation logic.
        """
        full_context_ids = self._context_ids
        full_prefix_len = self.prefix_length

        # Extract selected tokens for the 'fake' context
        relative_pos = (self._selected_positions - self.prefix_length).to("cpu")
        selected_ids = full_context_ids[:, self._selected_positions.to("cpu")]

        try:
            # Swap context to trick parent prepare() into generating prompts for selected tokens only
            self._context_ids = selected_ids
            self.prefix_length = 0

            # Parent does the heavy lifting (chunking + prompt creation)
            parent_pairs = super().prepare(model, tokenizer, chunk_size, prev_postfix_size)
        finally:
            # Always restore state
            self._context_ids = full_context_ids
            self.prefix_length = full_prefix_len

        # Map chunks back to their absolute positions using split
        chunk_lengths = [p[0].shape[1] for p in parent_pairs]
        pos_chunks = relative_pos.split(chunk_lengths)

        return [(pair[0], pair[1], pos) for pair, pos in zip(parent_pairs, pos_chunks)]

    def _perform_kvzip_compression(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """Execute Stage 1 (Selection) -> Stage 2 (Reconstruction) -> Compression."""
        self.context_length = self._context_ids.shape[1]

        # Stage 1
        inner_scores = self._compute_inner_press_scores(model)
        self._selected_positions = self._select_top_positions(inner_scores)

        logger.info(f"[KV²] Selected {len(self._selected_positions)} reconstruction queries.")

        # Stage 2
        for _, repeat_ids, chunk_positions in self.prepare(model, tokenizer):
            # Update state for score_kvzip
            self._current_chunk_positions = chunk_positions + self.prefix_length
            self.start_idx = self._current_chunk_positions[0].item()

            # Reconstruction pass
            model(
                input_ids=repeat_ids.to(model.device),
                past_key_values=self._cache,
                num_logits_to_keep=1,
            )

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
        Like KVzip, attends to the entire context; unlike KVzip, uses only selected queries.
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads, head_dim = module.config.num_attention_heads, module.head_dim
        num_kv_heads = module.config.num_key_value_heads

        # Prepare Queries
        queries = get_prerope_query_states(module, hidden_states)
        cos, sin = kwargs["position_embeddings"]
        queries = (queries * cos.unsqueeze(1)) + (rotate_half(queries) * sin.unsqueeze(1))
        queries = queries.view(bsz, num_kv_heads, num_heads // num_kv_heads, q_len, head_dim)

        # Prepare Keys: [Sink] + [All Context] + [Repeat Queries]
        sink = min(self.n_sink, self.start_idx)
        keys_subsampled = torch.cat(
            [
                keys[:, :, :sink],
                keys[:, :, sink : self.context_length],
                keys[:, :, -q_len:],
            ],
            dim=2,
        ).unsqueeze(2).transpose(-2, -1).contiguous()

        # Attention
        attn_weights = torch.matmul(queries, keys_subsampled) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Aggregate Scores (Max over queries/groups)
        scores = attn_weights[..., sink : self.context_length].amax(dim=(-3, -2))

        # Update global scoreboard using Max accumulation
        layer_idx = int(module.layer_idx)
        current_scores = self.score_val[layer_idx][..., sink : self.context_length]
        self.score_val[layer_idx][..., sink : self.context_length] = torch.maximum(current_scores, scores)

        # Return original keys/values (stripping the repeat chunk)
        return keys[:, :, : self.context_length], values[:, :, : self.context_length]
