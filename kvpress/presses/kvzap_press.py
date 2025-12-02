# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from dataclasses import dataclass

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
    config_class = KVzapConfig

    def __init__(self, config):
        super().__init__(config)
        if config.hidden_dim is None:
            # Linear model
            self.module_list = nn.ModuleList(
                [nn.Linear(config.input_dim, config.output_dim) for _ in range(config.n_modules)]
            )
        else:
            # 1-layer MLP model
            self.module_list = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(config.input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim, config.output_dim),
                )
                for _ in range(config.n_modules)
            )

    def forward(self, x):
        return torch.stack([module(x[:, i, :]) for i, module in enumerate(self.module_list)], dim=1)

@dataclass
class KVzapPress(ScorerPress):
    """
    KVzap approximates KVzip+ (an improved version of KVzip) by training a small auxiliary
    model on top of the hidden states (see train_kvzap.py).
    The KVzapPress is designed to be used in conjunction with the ThresholdPress
    """
    kvzap_model_name_or_path: str = None

    def __post_init__(self):
        assert self.kvzap_model_name_or_path is not None, "kvzap_model_name_or_path must be provided"
        self.kvzap_model = KVzapModel.from_pretrained(self.kvzap_model_name_or_path)

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        module = self.kvzap_model.module_list[module.layer_idx].to(hidden_states.device, dtype=hidden_states.dtype)
        scores = module(hidden_states).transpose(1, 2)
        return scores
