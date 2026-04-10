# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.content_adaptive_press import _apply_content_boosts, classify_content
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class ContentAdaptiveWrapper(BasePress):
    """Composable wrapper that adds content-type-aware scoring to any ScorerPress.

    Wraps an existing :class:`ScorerPress` and applies additive sink / recency
    boosts on top of its normalised scores.  This is designed as an *orthogonal*
    enhancement: if ``ContentAdaptiveWrapper(SnapKVPress)`` outperforms
    ``SnapKVPress`` **and** ``ContentAdaptiveWrapper(ExpectedAttentionPress)``
    outperforms ``ExpectedAttentionPress``, then content-type awareness is a
    general improvement layer, not a competing method.

    The architectural pattern follows :class:`AdaKVPress` — a ``BasePress``
    subclass that delegates scoring to a wrapped ``ScorerPress`` and applies
    additional logic on the scores before selecting tokens via top-k.

    Call :meth:`detect` before each sample to set the active content type,
    or assign :attr:`content_type` directly.

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method whose scores will be boosted.
    content_type : str, default="prose"
        Active content type.  One of ``"code"``, ``"math"``, ``"prose"``,
        ``"structured"``.
    """

    press: ScorerPress
    content_type: str = "prose"

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), (
            f"ContentAdaptiveWrapper requires a ScorerPress, got {type(self.press)}"
        )

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def detect(self, text: str) -> str:
        """Detect and store the content type from raw text.

        Returns the detected type for convenience.
        """
        self.content_type = classify_content(text)
        return self.content_type

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.press.compression_ratio == 0:
            return keys, values

        # Base scores from the wrapped press, normalised to [0, 1]
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        lo = scores.min(dim=-1, keepdim=True).values
        hi = scores.max(dim=-1, keepdim=True).values
        scores = (scores - lo) / (hi - lo + 1e-8)

        # Content-adaptive boosts
        scores = _apply_content_boosts(scores, self.content_type)

        # Top-k selection (mirrors ScorerPress.compress)
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.press.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
