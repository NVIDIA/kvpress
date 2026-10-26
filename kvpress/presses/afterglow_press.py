# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional

import torch

from kvpress.presses.scorer_press import ScorerPress


def afterglow(scores: torch.Tensor, gamma: float, n_sink: int, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    s'(t) = max(s(t), gate(t) * max_{n_sink <= j < t} gamma^(t - j) s(j)) along the last dimension, computed as one
    cumulative max in log space. Scores must be >= 0. The first n_sink tokens are not propagation sources.
    """
    if (scores < 0).any():
        raise ValueError("Afterglow requires scores >= 0, e.g. use scores.exp() for scorers with negative scores")
    positions = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.float32) * math.log(gamma)
    sources = scores.float().clamp_min(1e-30).log() - positions
    sources[..., :n_sink] = float("-inf")
    inherited = (torch.cummax(sources, dim=-1).values + positions).exp()
    if gate is not None:
        inherited = inherited * gate
    return torch.maximum(scores, inherited.to(scores.dtype))


class AfterglowPress(ScorerPress):
    """
    Afterglow: forward score propagation for query-aware compression.

    Query-aware scorers concentrate their scores on the first tokens of the span the question asks about, so the rest
    of the span is evicted. Afterglow lets each token inherit the decayed scores of the tokens before it, per layer and
    KV head: s'(t) = max(s(t), max_{j < t} gamma^(t - j) s(j)). Use it inside AdaKVPress or CriticalAdaKVPress, e.g.
    CriticalAdaKVPress(AfterglowPress(SnapKVPress())), as CriticalKV sets the scores it protects to the largest float.

    Based on Afterglow (https://www.alphaxiv.org/abs/2610.afterglow-query-aware-kv-cache-eviction).

    Parameters
    ----------
    press : ScorerPress
        The scorer whose scores are propagated. Its scores must be >= 0.
    gamma : float, default=0.95
        Decay of the propagated scores per token.
    n_sink : int, default=4
        Number of initial tokens (attention sinks) that are not propagation sources.
    """

    def __init__(self, press: ScorerPress, gamma: float = 0.95, n_sink: int = 4):
        self.press = press
        self.gamma = gamma
        self.n_sink = n_sink
        assert isinstance(self.press, ScorerPress), "AfterglowPress requires a ScorerPress as input"
        assert 0 < self.gamma < 1, "gamma must be in (0, 1)"

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property  # type: ignore[misc]
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        return afterglow(scores, self.gamma, self.n_sink)
