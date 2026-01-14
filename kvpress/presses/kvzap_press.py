# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from kvpress.presses.scorer_press import ScorerPress


class KVzapConfig(PretrainedConfig):
    model_type: str = "kvzap"
    input_dim: int
    output_dim: int
    hidden_dim: Optional[int] = None
    n_modules: int


class KVzapModel(PreTrainedModel):
    config_class = KVzapConfig  # type: ignore[assignment]

    def __init__(self, config):
        super().__init__(config)
        if config.hidden_dim is None:
            # Linear model
            self.layers = nn.ModuleList(
                [nn.Linear(config.input_dim, config.output_dim) for _ in range(config.n_modules)]
            )
        else:
            # 2-layer MLP model
            self.layers = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(config.input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim, config.output_dim),
                )
                for _ in range(config.n_modules)
            )

    def forward(self, x):
        return torch.stack([module(x[:, i, :]) for i, module in enumerate(self.layers)], dim=1)


@dataclass
class KVzapPress(ScorerPress):
    """
    KVzap (https://arxiv.org/abs/2601.07891) is a fast approximation of KVzip that works
    in both prefilling and decoding. It applies a lightweight surrogate model to the hidden
    states to predict importance scores for every KV pair.
    KVzapPress is designed to be used in conjunction with the ThresholdPress
    model_type can be "linear" or "mlp".
    """

    model_type: Literal["linear", "mlp"] = "mlp"
    kvzap_model_name: Optional[str] = field(default=None, init=False)

    def post_init_from_model(self, model):
        kvzap_model_name = f"nvidia/KVzap-{self.model_type}-{model.config.name_or_path.split('/')[-1]}"
        if kvzap_model_name != self.kvzap_model_name:
            self.kvzap_model_name = kvzap_model_name
            self.kvzap_model = KVzapModel.from_pretrained(self.kvzap_model_name)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        module = self.kvzap_model.layers[module.layer_idx]
        module = module.to(hidden_states.device, dtype=hidden_states.dtype).eval()
        with torch.no_grad():
            scores = module(hidden_states).transpose(1, 2)
        return scores
