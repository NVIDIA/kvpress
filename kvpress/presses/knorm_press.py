# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class KnormPress(ScorerPress):
    """
    Key norm-based KV cache compression.

    Prunes key-value pairs based on L2 norm of key vectors.
    Simple, efficient method requiring only norm calculation.

    Based on https://arxiv.org/pdf/2406.11430.

    Scores are computed in float32 (or float64 for float64 caches) so that low-precision
    caches do not tie the scores of distinct positions.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
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
        # Accumulate and return the norm in at least float32: a bfloat16 norm rounds the scores of
        # distinct positions onto the same value (~74% of per-head scores tie on Qwen3-8B), which
        # hands the choice of pruned pairs to top-k tie-breaking instead of the key norm. The keys
        # themselves are left in their original dtype.
        score_dtype = torch.promote_types(keys.dtype, torch.float32)
        return -torch.linalg.vector_norm(keys, dim=-1, dtype=score_dtype)
