# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Generator, Union


import numpy as np
import torch
import types
from torch import nn
from transformers.models.llama import LlamaForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)


class Aggregator(ABC):
    n: int
    data: torch.Tensor
    neutral: float
    device: str

    def __init__(self, n, device):
        self.n = n
        self.device = device
        self._init_data()

    def _init_data(self):
        self.data = torch.full((self.n,), self.neutral, device=self.device)

    def reset(self):
        self._init_data()

    def partial_fit(self, nd_data: Union[torch.Tensor, np.ndarray]):
        if isinstance(nd_data, np.ndarray):
            nd_data = torch.from_numpy(nd_data)
        if nd_data.ndim == 1:
            nd_data = nd_data.unsqueeze(0)
        return self._partial_fit(nd_data)

    @abstractmethod
    def _partial_fit(self, nd_data: torch.Tensor):
        pass

    def transform(self):
        return self.data

    def fit(self, *args):
        self._init_data()
        self.partial_fit(*args)

    def fit_transform(self, *args):
        self.fit(*args)
        return self.transform()


class MaxAggregator(Aggregator):
    def __init__(self, n, device):
        self.neutral = -torch.inf
        super().__init__(n, device)

    def _partial_fit(self, nd_data: torch.Tensor):
        new_max_data = nd_data.amax(dim=tuple(range(len(nd_data.shape)-1)))
        self.data = torch.maximum(self.data, new_max_data)


class MeanAggregator(Aggregator):
    sum_data: torch.Tensor
    count_data: torch.Tensor

    def __init__(self, n, device):
        self.neutral = 0.
        super().__init__(n, device)
        self.sum_data = torch.full((n, ), self.neutral, device=self.device)
        self.count_data = torch.full((n, ), self.neutral, device=self.device)

    def _partial_fit(self, nd_data: torch.Tensor):
        new_sum_data = nd_data.sum(dim=tuple(range(len(nd_data.shape)-1)))
        new_count_data = torch.ones_like(nd_data, device=self.device).sum(dim=tuple(range(len(nd_data.shape)-1)))
        self.sum_data += new_sum_data
        self.count_data += new_count_data
        self.data = self.sum_data / self.count_data


aggregator_by_name = {
    "mean": MeanAggregator,
    "max": MaxAggregator,
}


def get_aggregator(aggregator_name: str) -> type[Aggregator]:
    aggregator_name = aggregator_name.lower()
    if aggregator_name not in aggregator_by_name:
        raise ValueError(f"Unknown aggregator_name: {aggregator_name}. "
                         f"Available: {list(aggregator_by_name)}")
    return aggregator_by_name[aggregator_name]


def copy_cache(cache: DynamicCache):
    new_cache = DynamicCache()
    for (layer, (k, v)) in enumerate(cache):
        new_cache.update(layer_idx=layer, key_states=k.clone(), value_states=v.clone())
    return new_cache


@dataclass
class KVComposePress(BasePress):
    """
    KVComposePress implements KVCompose: a structured KV cache compression
    method that remains compatible with standard inference pipelines.

    Setting `structured=False` enables the unstructured variant where each head
    retains tokens independently (no composite alignment). This generally yields
    better theoretical performance but is incompatible with standard KV cache layouts
    unless the attention mechanism is modified.

    Based on KVCompose (https://arxiv.org/abs/2509.05165).

    Parameters
    ----------
    structured : bool, default=True
        Whether to use the structured or unstructured method.
    compression_ratio : float, default=0.0
        Global fraction of KV tokens to remove.
    agg_task : str, default="max"
        Strategy to form per-context-token importance score per layer/head from
        attention (e.g. 'max', 'mean').
    agg_group : str, default="mean"
        Aggregation within each head across groups (for grouped query attention).
    agg_head : str, default="mean"
        Aggregation across heads to form composite importance score (used for
        structured alignment).
    add_v_norm : bool, default=False
        Whether to multiply token score by the norm of its value vector.
    add_mean_across_heads : bool, default=True
        Whether to augment token scores with the mean score across all heads
        to improve stability.
    keep_token_lower_bound : int, default=0
        Minimum number of tokens to keep in each layer.
    """

    def __init__(self,
                 structured: bool = True,
                 compression_ratio: float = 0,
                 agg_task: str = "max",
                 agg_group: str = "mean",
                 agg_head: str = "mean",
                 add_v_norm: bool = False,
                 add_mean_across_heads: bool = True,
                 keep_token_lower_bound: int = 0,
                 ):
        self.structured = structured
        self.compression_ratio = compression_ratio
        self.agg_task = agg_task
        self.agg_group = agg_group
        self.agg_head = agg_head

        self.add_v_norm = add_v_norm
        self.add_mean_across_heads = add_mean_across_heads
        self.keep_token_lower_bound = keep_token_lower_bound

        super().__init__()

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"

    def _init_statistics(self):
        self.task_aggregators = [
            [get_aggregator(self.agg_task)(self.context_len, self.device) for _ in range(self.num_att_heads)]
            for _ in range(self.num_layers)
        ]

    def _register_model(self, model: PreTrainedModel):
        self.model = model
        self.num_layers: int = getattr(model.config, "num_hidden_layers")
        self.num_att_heads: int = getattr(model.config, "num_attention_heads")
        self.num_kv_heads: int = getattr(model.config, "num_key_value_heads")
        self.num_kv_groups: int = self.num_att_heads // self.num_kv_heads
        self.device = next(model.parameters()).device

    def register_context_ids(self, context_ids: torch.Tensor):
        self.context_ids = context_ids
        self.context_len = self.context_ids.shape[-1]
        self.prompt_ids: list[torch.Tensor] = []
        self._init_statistics()

    def register_prompt_ids(self, prompt_ids: list[torch.Tensor]):
        self.prompt_ids = prompt_ids

    def _register_cache(self, cache: DynamicCache):
        self.cache = cache

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Fitting self.task_aggregators with the attention scores from the forward pass.
        Attentions are cleaned up from the output to save memory.
        """
        layer = int(module.layer_idx)

        layer_attentions = output[1]
        assert layer_attentions is not None

        for att_head in range(self.num_att_heads):
            if layer_attentions.shape[3] == layer_attentions.shape[2]:
                # Skip self-to-self attention (prefill step), only record context-to-query attentions
                continue
            layer_att_head_attention = layer_attentions[:, att_head, :, :self.context_len]
            self.task_aggregators[layer][att_head].partial_fit(layer_att_head_attention)

        # Clean up attention to save memory.
        output = list(output)
        del output[1]
        output.append(None)

        return output

    def get_scores(self):
        """
        Obtaining token scores by doing aggregation over groups for every kv head.
        """
        self.scores = torch.zeros((self.num_layers, self.num_kv_heads, self.context_len), device=self.device)
        for layer in range(self.num_layers):
            for kv_head in range(self.num_kv_heads):
                group_aggregator = get_aggregator(self.agg_group)(self.context_len, self.device)
                for att_head in range(kv_head * self.num_kv_groups, (kv_head + 1) * self.num_kv_groups):
                    group_aggregator.partial_fit(self.task_aggregators[layer][att_head].transform())
                self.scores[layer, kv_head] = group_aggregator.transform()

    def enhance_scores(self):
        """
        Enhance token scores by incorporating value vector norms and mean scores.
        """
        for layer in range(self.num_layers):
            if self.add_v_norm:
                for kv_head in range(self.num_kv_heads):
                    v = self.cache.layers[layer].values[0, kv_head].detach()
                    self.scores[layer, kv_head] = self.scores[layer, kv_head] * v.norm(dim=1)
            if self.add_mean_across_heads:
                self.scores[layer] += self.scores[layer].mean(dim=0, keepdim=True)

    def get_composite_scores(self):
        self.composite_scores_per_head = self.scores.sort(dim=-1, descending=True)[0]

        self.composite_scores_per_layer = torch.full((self.num_layers, self.context_len), 0., device=self.device)
        for layer in range(self.num_layers):
            layer_aggregator = get_aggregator(self.agg_head)(self.context_len, self.device)
            for kv_head in range(self.num_kv_heads):
                layer_aggregator.partial_fit(self.scores[layer, kv_head].sort(descending=True)[0])
            self.composite_scores_per_layer[layer] = layer_aggregator.transform()

    def get_important_per_layer(self):
        """
        Calculates how many tokens to keep per layer (and per head for unstructured).
        """
        self.get_composite_scores()

        self.composite_scores_per_layer[..., :self.keep_token_lower_bound] += 1e9
        self.composite_scores_per_layer[0] = \
            self.composite_scores_per_layer.max(dim=0).values  # Ensures first layer is the largest.
        threshold_layer = self.composite_scores_per_layer.quantile(self.compression_ratio)
        self.important_per_layer = (self.composite_scores_per_layer >= threshold_layer).sum(dim=-1).cpu().numpy()

        self.composite_scores_per_head[..., :self.keep_token_lower_bound] += 1e9
        threshold_head = self.composite_scores_per_head.quantile(self.compression_ratio)
        self.important_per_head = (self.composite_scores_per_head >= threshold_head).sum(dim=-1).cpu().numpy()

    def prepare_important_masks(self):
        """
        Building masks of tokens to keep per kv head.
        """
        self.get_scores()
        self.enhance_scores()
        self.get_important_per_layer()

        self.important_mask_per_kv_head = [
            [
                torch.zeros(size=(self.context_len, ), device=self.device, dtype=torch.bool)
                for _ in range(self.num_kv_heads)
            ]
            for _ in range(self.num_layers)
        ]

        for layer in range(self.num_layers):
            for kv_head in range(self.num_kv_heads):
                count_of_important = (
                    self.important_per_layer[layer]
                    if self.structured
                    else self.important_per_head[layer, kv_head]
                )
                important_indices = torch.argsort(self.scores[layer, kv_head], descending=True)[:count_of_important]
                self.important_mask_per_kv_head[layer][kv_head][important_indices] = True

    def compress_structured(self) -> None:
        """
        Preparing compressed version of the cache.
        For KVPress, we modify the cache in-place (stored in self.cache).
        """
        for layer in range(self.num_layers):
            kv_over_layer: list[list[torch.Tensor]] = [[], []]
            for kv_head in range(self.num_kv_heads):
                important_mask = self.important_mask_per_kv_head[layer][kv_head]

                keys = self.cache.layers[layer].keys[0, kv_head][:self.context_len]
                values = self.cache.layers[layer].values[0, kv_head][:self.context_len]
                keys = keys[important_mask]
                values = values[important_mask]
                kv_over_layer[0].append(keys)
                kv_over_layer[1].append(values)
            new_key_states = torch.stack(kv_over_layer[0], dim=0).unsqueeze(0)
            new_value_states = torch.stack(kv_over_layer[1], dim=0).unsqueeze(0)

            self.cache.layers[layer].keys = new_key_states
            self.cache.layers[layer].values = new_value_states

    def compress_unstructured(self, model: PreTrainedModel) -> None:
        """
        Storing evicted indices in module.masked_key_indices.
        Relies on attention_patch.py implementation that simulates real eviction.
        Supports only batch size 1.
        """
        if self.context_ids.shape[0] != 1:
            raise NotImplementedError("Unstructured compression supports only batch size 1.")
        for layer_idx, layer in enumerate(model.model.layers):
            masked_over_layer: list[list[torch.Tensor]] = [[], [], []]

            for kv_head in range(self.num_kv_heads):
                non_important_mask = ~self.important_mask_per_kv_head[layer_idx][kv_head]
                num_non_important_tokens = int(non_important_mask.sum().item())
                batch_indices = torch.full((num_non_important_tokens, ), 0, device=self.device)
                head_indices = torch.full((num_non_important_tokens, ), kv_head, device=self.device)
                seq_indices = non_important_mask.nonzero(as_tuple=True)[0]
                masked_over_layer[0].append(batch_indices)
                masked_over_layer[1].append(head_indices)
                masked_over_layer[2].append(seq_indices)
            layer.self_attn.masked_key_indices = tuple(map(lambda x: torch.cat(x, dim=0), masked_over_layer))

    def compress_cache(self, model: PreTrainedModel) -> None:
        if self.structured:
            self.compress_structured()
        else:
            self.compress_unstructured(model)

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager to apply a compression method to a model.
        Apply this context manager during the pre-filling phase to compress the context.

        Parameters
        ----------
        model : PreTrainedModel
            Model to apply the compression method to
        """

        if not isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM)):
            logger.warning(f"Model {type(model)} not tested")

        self._register_model(model)

        def new_forward(self,
                        input_ids,
                        past_key_values,
                        *args,
                        press: KVComposePress,
                        **kwargs,
                        ):
            press.register_context_ids(input_ids)

            original_attn_implementation = self.model.config._attn_implementation
            self.model.config._attn_implementation = "eager"
            outputs = self.original_forward_KVComposePress(
                input_ids=input_ids,
                past_key_values=past_key_values,
                *args,
                **kwargs,
                )

            press._register_cache(past_key_values)
            for prompt_ids in (press.prompt_ids or [press.context_ids]):
                cache = past_key_values
                self.original_forward_KVComposePress(
                    input_ids=prompt_ids.to(self.model.device),
                    past_key_values=cache,
                    *args,
                    **kwargs,
                    )

            self.model.config._attn_implementation = original_attn_implementation
            return outputs

        hooks = []
        try:
            for layer in model.model.layers:
                layer.self_attn.rotary_emb = model.model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))

            setattr(model, "original_forward_KVComposePress", model.model.forward)
            new_forward_with_press = partial(new_forward, press=self)
            model.model.forward = types.MethodType(new_forward_with_press, model)

            yield
        finally:
            model.model.forward = getattr(model, "original_forward_KVComposePress")
            delattr(model, "original_forward_KVComposePress")
            for forward_hook in hooks:
                forward_hook.remove()
            self.prepare_important_masks()
            self.compress_cache(model)
