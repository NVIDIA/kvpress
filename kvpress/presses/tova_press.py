# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from kvpress.presses.snapkv_press import SnapKVPress


@dataclass
class TOVAPress(SnapKVPress):
    """
    TOVA (https://arxiv.org/abs/2401.06104) use the attention of the last token averaged across heads
    to estimate the importance of the previous KV pairs. This press was reviewed by Michael Hassid,
    one of the authors of the TOVA paper.
    """

    compression_ratio: float = 0.0
    window_size: int = 1  # re-use the attention weight computation from SnapKVPress for last token

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        if attentions is not None:
            attn_weights = attentions[..., -1:, :-1]
        else:
            attn_weights = self.compute_window_attention(module, hidden_states, keys)

        # Average across heads
        scores = attn_weights.mean(1)
        scores = scores.repeat(1, keys.shape[1], 1)

        # Keep the last token. Very slight difference from TOVA but last attention weight is usually very high
        scores = F.pad(scores, (0, 1), value=scores.max().item())

        return scores
