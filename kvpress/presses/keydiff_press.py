# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class KeyDiffPress(ScorerPress):
    """
    KeyDiff (https://arxiv.org/abs/2504.15364) evict tokens based solely on key similarity.
    """
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
