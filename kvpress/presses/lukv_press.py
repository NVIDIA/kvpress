# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel

from kvpress.presses.base_press import BasePress
from kvpress.presses.expected_attention_press import ExpectedAttentionPress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)

LLAMA_3_1_8B_EA_CURVE_URL = (
    "https://raw.githubusercontent.com/baidu-baige/LU-KV/main/"
    "evaluation/curve_data/llama-3.1-8b/ea_0.02_sink4_win1_llama_avg_ratio.npy"
)

DEFAULT_BUDGET_CURVE_URLS = {
    "llama-3.1-8b": LLAMA_3_1_8B_EA_CURVE_URL,
}


@dataclass
class LUKVPress(BasePress):
    """
    LU-KV: head-wise budget allocation around a score-based press.

    LU-KV wraps a ``ScorerPress`` and uses a pre-computed budget curve to allocate
    different token budgets to each attention layer and KV head. The default
    configuration is ``LUKVPress(ExpectedAttentionPress(epsilon=2e-2), sink=4, window=1)``.
    Based on Predicting Future Utility: Global Combinatorial Optimization for
    Task-Agnostic KV Cache Eviction (https://arxiv.org/abs/2602.08585).

    Parameters
    ----------
    press : ScorerPress, default=ExpectedAttentionPress(epsilon=2e-2)
        The scoring method used to rank cached tokens within each KV head.
    compression_ratio : float, optional
        Fraction of KV pairs to remove globally. This value is delegated to the
        wrapped scorer.
    budget_curve_url : str, optional
        URL to a ``.npy`` budget curve with shape ``[99, num_layers, num_kv_heads]``.
        When omitted, a default Llama-3.1-8B expected-attention curve is selected
        from the model name. If no matching curve is available, LU-KV falls back
        to uniform layer/head budgets and emits a warning.
    cache_dir : str, optional
        Directory used to cache downloaded budget curves. Defaults to
        ``$XDG_CACHE_HOME/kvpress/lukv`` or ``~/.cache/kvpress/lukv``.
    sink : int, default=4
        Number of initial tokens to protect from eviction.
    window : int, default=1
        Number of most recent tokens to protect from eviction.
    """

    press: ScorerPress = field(default_factory=lambda: ExpectedAttentionPress(epsilon=2e-2))
    budget_curve_url: Optional[str] = None
    cache_dir: Optional[str] = None
    sink: int = 4
    window: int = 1

    _budget_curves: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    _global_kept_tokens: int = field(init=False, repr=False, default=0)
    _global_total_tokens: int = field(init=False, repr=False, default=0)

    def __init__(
        self,
        press: Optional[ScorerPress] = None,
        compression_ratio: Optional[float] = None,
        budget_curve_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
        sink: int = 4,
        window: int = 1,
    ):
        self.press = press if press is not None else ExpectedAttentionPress(epsilon=2e-2)
        if compression_ratio is not None:
            self.press.compression_ratio = compression_ratio
        self.budget_curve_url = budget_curve_url
        self.cache_dir = cache_dir
        self.sink = sink
        self.window = window
        self._budget_curves = None
        self._global_kept_tokens = 0
        self._global_total_tokens = 0
        self.__post_init__()

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "LUKVPress requires a ScorerPress as input"
        assert self.sink >= 0, "sink must be non-negative"
        assert self.window >= 0, "window must be non-negative"

    @property
    def compression_ratio(self) -> float:
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value: float):
        self.press.compression_ratio = value

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        self._global_kept_tokens = 0
        self._global_total_tokens = 0
        with super().__call__(model):
            yield

    def post_init_from_model(self, model: PreTrainedModel):
        self.press.post_init_from_model(model)
        if self._budget_curves is None:
            model_name = getattr(model.config, "name_or_path", "") or model.__class__.__name__
            budget_curve_url = self.budget_curve_url
            if budget_curve_url is None:
                if not self._supports_default_budget_curve():
                    self._use_uniform_budget_curve(
                        model,
                        "No default LU-KV budget curve is available for this press configuration.",
                    )
                    return
                budget_curve_url = self._default_budget_curve_url(model_name)

            if budget_curve_url is None:
                self._use_uniform_budget_curve(
                    model,
                    f"No default LU-KV budget curve is available for model '{model_name}'.",
                )
                return

            try:
                budget_curve_path = self._download_budget_curve(budget_curve_url)
                self._budget_curves = np.load(budget_curve_path)
                self._validate_budget_curves(model)
            except Exception as exc:
                self._use_uniform_budget_curve(
                    model,
                    f"Failed to load LU-KV budget curve from {budget_curve_url}: {exc}",
                )

    @staticmethod
    def _default_budget_curve_url(model_name: str) -> Optional[str]:
        normalized_model_name = model_name.lower()
        for name_fragment, url in DEFAULT_BUDGET_CURVE_URLS.items():
            if name_fragment in normalized_model_name:
                return url
        return None

    def _supports_default_budget_curve(self) -> bool:
        if not isinstance(self.press, ExpectedAttentionPress):
            return False
        return math.isclose(self.press.epsilon, 2e-2) and self.sink == 4 and self.window == 1

    def _model_curve_shape(self, model: PreTrainedModel) -> tuple[int, int]:
        num_layers = getattr(model.config, "num_hidden_layers", None)
        num_key_value_heads = getattr(model.config, "num_key_value_heads", None)
        if num_key_value_heads is None:
            num_key_value_heads = getattr(model.config, "num_attention_heads", None)
        if num_layers is None or num_key_value_heads is None:
            raise ValueError("LU-KV requires num_hidden_layers and num_key_value_heads or num_attention_heads.")
        return int(num_layers), int(num_key_value_heads)

    def _use_uniform_budget_curve(self, model: PreTrainedModel, reason: str):
        num_layers, num_key_value_heads = self._model_curve_shape(model)
        uniform_prune_ratios = np.arange(1, 100, dtype=np.float32).reshape(99, 1, 1) / 100
        self._budget_curves = np.broadcast_to(
            uniform_prune_ratios,
            (99, num_layers, num_key_value_heads),
        ).copy()
        logger.warning("%s Falling back to uniform LU-KV budgets.", reason)

    def _cache_root(self) -> Path:
        if self.cache_dir is not None:
            return Path(self.cache_dir).expanduser()
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        base_dir = Path(xdg_cache_home).expanduser() if xdg_cache_home else Path.home() / ".cache"
        return base_dir / "kvpress" / "lukv"

    def _download_budget_curve(self, url: str) -> Path:
        cache_root = self._cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)

        parsed_path = Path(urlparse(url).path)
        suffix = parsed_path.suffix or ".npy"
        stem = parsed_path.stem or "budget_curve"
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        cache_path = cache_root / f"{stem}-{url_hash}{suffix}"
        if cache_path.exists():
            return cache_path

        tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        try:
            urlretrieve(url, tmp_path)
            tmp_path.replace(cache_path)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise OSError(f"Failed to download LU-KV budget curve from {url}: {exc}") from exc
        return cache_path

    def _validate_budget_curves(self, model: PreTrainedModel):
        if self._budget_curves is None:
            raise ValueError("LU-KV budget curves are not loaded.")
        if self._budget_curves.ndim != 3 or self._budget_curves.shape[0] < 99:
            raise ValueError(
                "LU-KV budget curve must have shape [99, num_layers, num_kv_heads], "
                f"got {self._budget_curves.shape}."
            )

        num_layers = getattr(model.config, "num_hidden_layers", None)
        num_key_value_heads = getattr(model.config, "num_key_value_heads", None)
        if num_key_value_heads is None:
            num_key_value_heads = getattr(model.config, "num_attention_heads", None)

        if num_layers is not None and self._budget_curves.shape[1] != num_layers:
            raise ValueError(
                "LU-KV budget curve layer count does not match the model: "
                f"curve has {self._budget_curves.shape[1]}, model has {num_layers}."
            )
        if num_key_value_heads is not None and self._budget_curves.shape[2] != num_key_value_heads:
            raise ValueError(
                "LU-KV budget curve KV-head count does not match the model: "
                f"curve has {self._budget_curves.shape[2]}, model has {num_key_value_heads}."
            )

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
        layer_idx = module.layer_idx
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

        self._global_kept_tokens += int(final_keep_per_head.sum().item()) * bsz
        self._global_total_tokens += seq_len * num_key_value_heads * bsz
        total_layers = getattr(module.config, "num_hidden_layers", None)
        if total_layers is not None and layer_idx == total_layers - 1 and self._global_total_tokens > 0:
            keep_ratio = self._global_kept_tokens / self._global_total_tokens
            logger.info("[LUKVPress] Final global keep ratio: %.6f", keep_ratio)

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
