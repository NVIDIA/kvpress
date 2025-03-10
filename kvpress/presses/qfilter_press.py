# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress

from huggingface_hub import PyTorchModelHubMixin, get_collection

from typing import Generator

from transformers import (
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    PreTrainedModel,
    Qwen2ForCausalLM,
)

import logging
logger = logging.getLogger(__name__)

QFILTERS_COLLECTION = "nthngdy/q-filters-67a4994dcb302a3d37f3d119"


def get_model_name(model_id: str) -> str:
    """Extract model name from model_id."""
    model_name = model_id.split("/")[-1].replace("_qfilt", "")
    return model_name


def check_q_filters_available(model_name: str) -> bool:
    """Check if pre-trained QFilters are available for download for a specific model."""
    collection = get_collection(QFILTERS_COLLECTION)
    available_filters = {get_model_name(x.item_id) for x in collection.items}
    return model_name in available_filters


class QFilters(torch.nn.Module, PyTorchModelHubMixin):
    """Learnable Q-filters for KV cache pruning. (https://arxiv.org/abs/2503.02812)"""
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        kv_head_dim: int,
        model_name: str = None
    ):
        super().__init__()
        self.q_filters = torch.nn.Parameter(torch.randn(num_layers, num_kv_heads, kv_head_dim))
        self.model_name = get_model_name(model_name) if model_name else None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # we always keep a copy of the model name in the filters
        filters = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        filters.model_name = get_model_name(pretrained_model_name_or_path)
        return filters

    # representation of this object
    def __repr__(self):
        return f"QFilters(model_name={self.model_name})"


@dataclass
class QFilterPress(ScorerPress):
    """Prune KV pairs with Q-filters"""
    _q_filters: QFilters = None
    model_id: str = None

    def __post_init__(self):
        if self._q_filters is None and self.model_id is not None:
            model_name = get_model_name(self.model_id)
            self._load_filters_for_model(model_name)
        elif self._q_filters is not None and self.model_id is not None:
            # make sure that q_filters and model_id are consistent
            model_name = get_model_name(self.model_id)
            if self._q_filters.model_name != model_name:
                logger.warning(f"Using QFilters from {self._q_filters.model_name} for {model_name}.")

    @property
    def q_filters(self) -> QFilters:
        if self._q_filters is None:
            raise ValueError(
                "QFilters are not initialized. Either:\n"
                "1. Provide a model_id at initialization to download Qfilters automatically\n"
                "2. Download QFilterPress and set them with press.qfilters = your_filters\n"
            )
        return self._q_filters

    @q_filters.setter
    def q_filters(self, value: QFilters):
        self._q_filters = value

    def _load_filters_for_model(self, model_name: str) -> None:
        """Load filters for a specific model from the hub."""
        logger.info(f"Loading QFilters for {model_name}")
        try:
            self._q_filters = QFilters.from_pretrained(f"nthngdy/{model_name}_qfilt")
        except Exception as e:
            if not check_q_filters_available(model_name):
                logger.error(
                    f"QFilters not available for {model_name}. To use QFilters from another\
                         model, set q_filters manually."
                )
            else:
                logger.error(f"Unable to load QFilters for {model_name}: {e}")

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute a tensor of scores with shape (bsz, num_key_value_heads, q_len)
        The KV pairs with lowest scores will be pruned in the `compress` method.
        """
        # Get filter for this layer and match device/dtype
        layer_filter = self.q_filters.q_filters[module.layer_idx].to(
            device=keys.device,
            dtype=keys.dtype
        )

        # Compute score by projecting keys onto the filter
        scores = -(layer_filter[None, :, None] * keys).sum(dim=-1)
        return scores

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager to apply a compression method to a model.
        Apply this context manager during the pre-filling phase to compress the context.

        Parameters
        ----------
        model : PreTrainedModel
            Model to apply the compression method to

        Raises
        ------
        ValueError
            If unable to infer model type when q_filters is None
        """

        if not isinstance(model, (LlamaForCausalLM, MistralForCausalLM, Phi3ForCausalLM, Qwen2ForCausalLM)):
            logger.warning(f"Model {type(model)} not tested")

        if self._q_filters is None:
            try:
                model_name = get_model_name(model.config.name_or_path)
                self._load_filters_for_model(model_name)
            except AttributeError:
                raise ValueError("Unable to infer model type. Please provide path to download Qfilters.")

        model_name = get_model_name(model.config.name_or_path)
        if self.q_filters.model_name != model_name:
            logger.warning(f"Using QFilters from {self.q_filters.model_name} for {model_name}.")

        hooks = []
        try:
            for layer in model.model.layers:
                layer.self_attn.rotary_emb = model.model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()


"""
Example usage:

model_id = "meta-llama/Llama-3.2-1B"
unit_test_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).eval().cuda()

# Option 1: automatically download QFilters for the model on press call
press = QFilterPress(compression_ratio=0.2)
with press(model):
    # run inference with the model
    ...


# Option 2: provide model_id to download QFilters before press call
press = QFilterPress(model_id="meta-llama/Llama-3.2-1B")
with press:
    # run inference with the model
    ...


# Option 3: use pre-loaded QFilters
q_filters = QFilters.from_pretrained("nthngdy/llama-3.2-1b_qfilt")
press = QFilterPress()
press.q_filters = q_filters
with press:
    # run inference with the model
    ...
"""
