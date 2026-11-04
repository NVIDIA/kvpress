# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import numpy as np
import requests  # type: ignore[import-untyped]
import torch
from cachetools import LRUCache, cached  # type: ignore[import-untyped]
from torch import nn
from transformers import PreTrainedModel

from kvpress.presses.base_press import BasePress
from kvpress.presses.expected_attention_press import ExpectedAttentionPress
from kvpress.presses.scorer_press import ScorerPress

LLAMA_3_1_8B_EA_CURVE_URL = (
    "https://raw.githubusercontent.com/baidu-baige/LU-KV/main/"
    "evaluation/curve_data/llama-3.1-8b/ea_0.02_sink4_win1_llama_avg_ratio.npy"
)

BUDGET_CURVE_URLS = {
    # ExpectedAttentionPress(epsilon=2e-2), sink=4, window=1
    ("meta-llama/Llama-3.1-8B-Instruct", "ExpectedAttentionPress"): LLAMA_3_1_8B_EA_CURVE_URL,
}

cache = LRUCache(maxsize=128)


@cached(cache, key=lambda url: url)
def load_budget_curve(url: str) -> np.ndarray:
    response = requests.get(url)
    response.raise_for_status()
    return np.load(BytesIO(response.content), allow_pickle=False)


@dataclass
class LUKVPress(BasePress):
    """
    LU-KV: head-wise budget allocation around a score-based press.

    LU-KV wraps a ``ScorerPress`` and uses a pre-computed budget curve to allocate
    different token budgets to each attention layer and KV head. The default
    configuration is ``LUKVPress(ExpectedAttentionPress(epsilon=2e-2), sink=4, window=1)``.
    Based on Predicting Future Utility: Global Combinatorial Optimization for
    Task-Agnostic KV Cache Eviction (https://arxiv.org/abs/2602.08585).

    Budget curves are model- and scorer-specific. To add a new curve, upload the
    ``.npy`` file with shape ``[99, num_layers, num_kv_heads]`` and add an entry
    to ``BUDGET_CURVE_URLS`` keyed by ``(model.config.name_or_path, ScorerPressName)``.
    The default curve is for ``ExpectedAttentionPress(epsilon=2e-2), sink=4, window=1``
    on Llama-3.1-8B.

    New budget curves can be generated with the official LU-KV repository:
    https://github.com/baidu-baige/LU-KV. In that repository, run
    ``cd evaluation && bash curve_data/generate_curve.sh`` after setting
    ``MODEL_TYPE``, ``MODEL_PATH``, ``DATASET_PATH``, ``METHOD_CONFIGS``, and
    ``LAYERWISE_FLAG`` in ``evaluation/curve_data/generate_curve.sh``. The script
    first extracts per-context scorer statistics with ``step1_<model>.py`` and
    then computes the averaged layer/head budget curve with
    ``step2_compute_curve.py``.

    Parameters
    ----------
    press : ScorerPress, default=ExpectedAttentionPress(epsilon=2e-2)
        The scoring method used to rank cached tokens within each KV head.
    compression_ratio : float, default=0.0
        Fraction of KV pairs to remove globally.
    sink : int, default=4
        Number of initial tokens to protect from eviction.
    window : int, default=1
        Number of most recent tokens to protect from eviction.
    """

    press: ScorerPress = field(default_factory=lambda: ExpectedAttentionPress(epsilon=2e-2))
    compression_ratio: float = 0.0
    sink: int = 4
    window: int = 1

    _budget_curves: Optional[np.ndarray] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "LUKVPress requires a ScorerPress as input"
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        assert self.sink >= 0, "sink must be non-negative"
        assert self.window >= 0, "window must be non-negative"

    def post_init_from_model(self, model: PreTrainedModel):
        self.press.post_init_from_model(model)
        if self._budget_curves is None:
            self._budget_curves = self._load_budget_curves(model)

    def _load_budget_curves(self, model: PreTrainedModel) -> np.ndarray:
        model_name = model.config.name_or_path
        press_name = type(self.press).__name__
        url = BUDGET_CURVE_URLS.get((model_name, press_name))
        if url is None:
            raise KeyError(
                f"No LU-KV budget curve found for model={model_name!r} and press={press_name!r}. "
                f"Available curves: {list(BUDGET_CURVE_URLS.keys())}."
            )

        budget_curves = load_budget_curve(url)
        num_layers, num_key_value_heads = self._model_curve_shape(model)
        expected_shape = (99, num_layers, num_key_value_heads)
        if budget_curves.shape != expected_shape:
            raise ValueError(f"LU-KV budget curve must have shape {expected_shape}, got {budget_curves.shape}.")

        return budget_curves

    def _model_curve_shape(self, model: PreTrainedModel) -> tuple[int, int]:
        num_layers = getattr(model.config, "num_hidden_layers", None)
        num_key_value_heads = getattr(model.config, "num_key_value_heads", None)
        if num_key_value_heads is None:
            num_key_value_heads = getattr(model.config, "num_attention_heads", None)
        if num_layers is None or num_key_value_heads is None:
            raise ValueError("LU-KV requires num_hidden_layers and num_key_value_heads or num_attention_heads.")
        return int(num_layers), int(num_key_value_heads)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio <= 0:
            return keys, values
        if self._budget_curves is None:
            raise ValueError("LU-KV budget curves are not loaded. Use LUKVPress as a model context manager first.")
        assert module.config._attn_implementation != "eager", "eager mode not supported"

        bsz, num_key_value_heads, seq_len, _ = keys.shape
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        protected_score = scores.max().item() + 1
        if self.sink > 0:
            safe_sink = min(self.sink, seq_len)
            scores[..., :safe_sink] = protected_score
        if self.window > 0:
            window_start = max(0, seq_len - self.window)
            scores[..., window_start:] = protected_score

        target_idx = int(round(self.compression_ratio * 100)) - 1
        target_idx = max(0, min(98, target_idx))
        layer_idx = int(getattr(module, "layer_idx"))
        try:
            local_prune_ratios = torch.as_tensor(
                self._budget_curves[target_idx, layer_idx],
                device=keys.device,
                dtype=torch.float32,
            )
        except IndexError as exc:
            raise ValueError(
                f"LU-KV budget curve does not contain target index {target_idx} and layer {layer_idx}."
            ) from exc

        if local_prune_ratios.shape[0] != num_key_value_heads:
            raise ValueError(
                "LU-KV budget curve KV-head count does not match current keys: "
                f"curve has {local_prune_ratios.shape[0]}, keys have {num_key_value_heads}."
            )

        head_keep_rates = (1.0 - local_prune_ratios).clamp(min=0.0, max=1.0)
        ideal_keep_counts = head_keep_rates * seq_len
        total_keep_target = int(torch.round(ideal_keep_counts.sum()).item())
        total_keep_target = max(num_key_value_heads, min(num_key_value_heads * seq_len, total_keep_target))
        base_keep_counts = torch.floor(ideal_keep_counts).long()
        remainder = total_keep_target - int(base_keep_counts.sum().item())

        if remainder > 0:
            fractional_parts = ideal_keep_counts - base_keep_counts
            top_k_indices = torch.topk(fractional_parts, k=min(remainder, num_key_value_heads)).indices
            base_keep_counts[top_k_indices] += 1

        final_keep_per_head = base_keep_counts.clamp_(min=1, max=seq_len)
        num_to_prune_per_head = seq_len - final_keep_per_head

        if torch.all(num_to_prune_per_head <= 0):
            module.masked_key_indices = None
            return keys, values

        sorted_indices = torch.argsort(scores, dim=-1, descending=True, stable=True)
        rank = torch.arange(seq_len, device=scores.device).view(1, 1, seq_len).expand_as(sorted_indices)
        keep_mask = rank < final_keep_per_head.view(1, num_key_value_heads, 1)
        prune_mask = ~keep_mask

        batch_indices, head_indices, rank_indices = torch.where(prune_mask)
        pruned_seq_indices = sorted_indices[batch_indices, head_indices, rank_indices]
        module.masked_key_indices = (batch_indices, head_indices, pruned_seq_indices)  # type: ignore[assignment]

        return keys, values
