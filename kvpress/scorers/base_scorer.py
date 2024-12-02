# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import nn


class BasesScorer:
    """Base class for scorers"""

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """Compute the scores for each KV pair in the layer.

        Parameters
        ----------
        module :
            Transformer layer, see `hook` method for more details.
        hidden_states :
            Hidden states of the layer.
        keys :
            Keys of the cache. Note keys are after RoPE.
        values :
            Values of the cache.
        attentions :
            Attention weights of the layer.
        kwargs :
            Keyword arguments, as given to the forward pass of the layer.

        Returns
        -------
            Scores for each KV pair in the layer, shape keys.shape[:-1].

        """
        raise NotImplementedError
