# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.leverage_press import LeverageScorePress
from kvpress.presses.non_causal_attention_press import NonCausalAttnPress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class CompactorPress(ScorerPress):
    """
    Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores

    Compactor blends: geometry-based outlier scores via (approximate) statistical leverage on key
    embeddings; and non-causal, chunked attention. Currently only supports prefill. The presented
    version slightly differs from the paper in that: (1) we set blending=compression_ratio by default,
    which is a good heuristic and should work for most users and (2) we use a cholesky
    decomposition to compute the leverage scores. Please see the paper for an in-depth discussion. The
    press is implemented as a wrapper that combines ``NonCausalAttnPress`` and
    ``LeverageScorePress`` scores.

    References:
    - Chari & Van Durme (2025): "Compactor: Calibrated Query-Agnostic KV Cache
      Compression with Approximate Leverage Scores" (https://arxiv.org/pdf/2507.08143v1)

    Parameters
    ----------
    compression_ratio : float, default ``0.0``
         Fraction of key-value pairs to remove during compression.
    sink_size_start : int, default ``8``
        Number of initial sink tokens to always protect.
    sink_size_end : int, default ``4``
        Number of most-recent tokens to always protect.
    chunk_size : int, default ``256``
        Chunk size used to in non-causal attention.
    sketch_dimension: int, default ``48``
        Size of Gaussian sketch.
    blending : Optional[float], default ``None``
        Weight for blending scores in the final output. If ``None``,
        it set to ``compression_ratio``, which tends to be a good default.

    Output
    ------
    score(...) returns a tensor of shape (B, H_kv, S) with higher values
    indicating more important tokens for retention.
    """

    sink_size_start: int = 8
    sink_size_end: int = 4
    chunk_size: int = 256
    sketch_dimension: int = 48
    blending: Optional[float] = None

    _leverage_press: Optional[LeverageScorePress] = None
    _non_causal_press: Optional[NonCausalAttnPress] = None

    def __post_init__(self):
        # build child presses if not provided
        self._leverage_press = LeverageScorePress(
            compression_ratio=self.compression_ratio, sketch_dimension=self.sketch_dimension
        )
        self._non_causal_press = NonCausalAttnPress(
            compression_ratio=self.compression_ratio, chunk_size=self.chunk_size
        )

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "compression_ratio":
            if "_leverage_press" in self.__dict__:
                self._leverage_press.compression_ratio = value
            if "_non_causal_press" in self.__dict__:
                self._non_causal_press.compression_ratio = value
        if name == "sketch_dimension":
            if "_leverage_press" in self.__dict__:
                self._leverage_press.sketch_dimension = value
        if name == "chunk_size":
            if "_non_causal_press" in self.__dict__:
                self._non_causal_press.chunk_size = value

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """Blend leverage and non-causal attention into final importance scores"""
        n_queries = hidden_states.shape[-2]
        assert keys.shape[-2] == n_queries, "CompactorPress only supports prefill at the moment"
        left_keep = min(self.sink_size_start, n_queries)
        right_keep = min(self.sink_size_end, max(0, n_queries - left_keep))
        start_idx, end_idx = left_keep, (None if right_keep == 0 else -right_keep)

        hs = hidden_states[:, start_idx:end_idx]
        keys = keys[..., start_idx:end_idx, :]
        values = values[..., start_idx:end_idx, :]
        cos, sin = kwargs["position_embeddings"]
        sliced_kwargs = {"position_embeddings": (cos[..., start_idx:end_idx, :], sin[..., start_idx:end_idx, :])}

        l_scores = self._leverage_press.score(
            module=module, hidden_states=hs, keys=keys, values=values, attentions=attentions, kwargs=sliced_kwargs
        )
        attn_scores = self._non_causal_press.score(
            module=module, hidden_states=hs, keys=keys, values=values, attentions=attentions, kwargs=sliced_kwargs
        )
        # sanity check. this breaks when not in prefill
        assert attn_scores.shape == l_scores.shape, "CompactorPress only supports prefill at the moment"
        blending = self.blending if self.blending is not None else self.compression_ratio
        blending = 0.35 if blending is None else blending
        scores = blending * l_scores + attn_scores
        # protect sinks by padding
        scores = F.pad(scores, (left_keep, right_keep), value=scores.detach().max())
        return scores
