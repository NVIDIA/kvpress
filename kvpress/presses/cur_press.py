# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class CURPress(ScorerPress):
    """
    Press based on `CurDKV` (https://arxiv.org/abs/2509.15038) which computes approximate leverage scores
    for keys (k2) and values (v2) and combines them to prune the KV cache.

    If `use_random_leverage` is true (default is False), keys and values are first
    multiplied by a random projection matrix G.
    If `use_local_approximation` is true (default), the scores are averaged over a
    local window of size `local_window_size`.
    Depending on `leverage_type`, returns either k2, v2, (k2 + v2) / 2, or k2 * v2 (default)
    Finally, the first `num_sinks` tokens are set to 1.0 to preserve some initial "attention sinks".
    """

    compression_ratio: float
    num_sinks: int = 4
    leverage_type: Literal["key", "value", "kv_avg", "kv_product"] = "kv_product"
    use_random_leverage: bool = False
    use_local_approximation: bool = True
    local_window_size: int = 16

    def score(self, module, hidden_states, keys, values, attentions, kwargs):

        if self.use_random_leverage:
            r = 20
            G = torch.randn(keys.shape[-1], r, device=keys.device) / math.sqrt(r)
            keys = keys @ G
            values = values @ G

        k2 = (keys**2).sum(dim=-1)
        v2 = (values**2).sum(dim=-1)

        if self.use_local_approximation:
            b, h, n = k2.shape
            w = self.local_window_size
            k2 = F.pad(k2, (0, (w - n % w) % w)).reshape(b, h, -1, w)
            k2 = (k2 / k2.sum(dim=-1, keepdim=True)).reshape(b, h, -1)[:, :, :n]
            v2 = F.pad(v2, (0, (w - n % w) % w)).reshape(b, h, -1, w)
            v2 = (v2 / v2.sum(dim=-1, keepdim=True)).reshape(b, h, -1)[:, :, :n]

        if self.leverage_type == "key":
            scores = k2
        elif self.leverage_type == "value":
            scores = v2
        elif self.leverage_type == "kv_avg":
            scores = (k2 + v2) / 2
        elif self.leverage_type == "kv_product":
            scores = k2 * v2
        else:
            raise ValueError("Unknown leverage type: choose from 'kv_avg', 'key', 'value' or 'kv_product'")

        scores /= scores.sum(dim=-1, keepdim=True)
        scores[:, :, : self.num_sinks] = 1.0

        return scores
