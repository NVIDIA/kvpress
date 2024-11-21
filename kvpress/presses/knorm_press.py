# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import nn

from kvpress.presses.base_press import BasePress


class KnormPress(BasePress):
    """Prune KV pairs with highest L2 norm of keys (https://arxiv.org/pdf/2406.11430)"""

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        return -keys.norm(dim=-1)
