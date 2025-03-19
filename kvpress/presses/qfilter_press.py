# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cache
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from huggingface_hub import PyTorchModelHubMixin, get_collection

from kvpress.presses.scorer_press import ScorerPress


class QFilters(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_layers: int, num_kv_heads: int, kv_head_dim: int):
        super().__init__()
        self.q_filters = torch.nn.Parameter(torch.randn(num_layers, num_kv_heads, kv_head_dim))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        return super().from_pretrained(pretrained_model_name_or_path)


@dataclass
class QFilterPress(ScorerPress):
    """
    Prune KV pairs with Q-filters
    """

    def __post_init_from_model__(self, model):
        model_name = model.config.name_or_path.split("/")[-1]
        self.q_filters = self.load_q_filters(model_name)
        self.q_filters = self.q_filters.to(model.dtype)

    @staticmethod
    @cache
    def load_q_filters(model_name):
        try:
            return QFilters.from_pretrained(f"nthngdy/{model_name}_qfilt").q_filters
        except TypeError:
            raise ValueError(
                f"Could not load Q-filters for {model_name}. Available models: {QFilterPress.available_qfilters()}"
            )

    @staticmethod
    def available_qfilters():
        collection = get_collection("nthngdy/q-filters-67a4994dcb302a3d37f3d119", token=False)
        return [x.item_id.split("/")[-1][:-6] for x in collection.items]

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        q_filter = self.q_filters[module.layer_idx][None, :, None]
        q_filter = q_filter.to(keys.device)
        scores = -(q_filter * keys).sum(dim=-1)
        return scores

    @contextmanager
    def __call__(self, model):
        self.__post_init_from_model__(model)
        with super().__call__(model):
            yield
