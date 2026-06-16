# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch
from torch import nn
from transformers import PreTrainedModel, QuantizedCache

from kvpress.presses.base_press import SUPPORTED_MODELS, BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.snapkv_press import SnapKVPress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class ZigZagKVPress(BasePress):
    """
    Dynamic per-layer KV cache compression based on attention concentration.

    Wrapper press that allocates per-layer KV cache budgets based on each layer's
    uncertainty, measured via LMBA (Layer Minimum Budget to maintain Attention).
    Layers with diffuse attention (high LMBA) receive larger budgets; layers with
    concentrated attention (low LMBA) receive smaller budgets. The total budget is
    conserved so the average compression ratio matches ``compression_ratio``.

    Based on ZigZagKV (https://arxiv.org/abs/2412.09036, COLING 2025).

    Unlike a naive implementation, this press is flash-attention compatible: it does
    not require ``attn_implementation="eager"``. Importance scores (and LMBA) are
    derived from the inner press's score function, which for SnapKV-style presses
    recomputes a small observation-window attention internally. Compression is
    applied with a single lightweight deferral: per-layer LMBA scalars and score
    tensors are collected during the forward pass, the global budget allocation is
    computed once all layers are seen, and each layer is then physically resized
    (genuine per-layer cache sizes, no padding).

    Parameters
    ----------
    press : ScorerPress, default=SnapKVPress()
        The underlying scoring method used to rank tokens within each layer.
        An attention-based press such as SnapKVPress is recommended so that LMBA
        reflects the attention distribution.
    compression_ratio : float, default=0.0
        Target average fraction of tokens to remove across all layers.
    attention_threshold : float, default=0.9
        Fraction of cumulative score mass used to determine LMBA per layer.
    b_bound_ratio : float, default=0.2
        Minimum per-layer budget as a fraction of the average budget. Prevents any
        single layer from being starved of tokens.
    """

    press: ScorerPress = field(default_factory=SnapKVPress)
    compression_ratio: float = 0.0
    attention_threshold: float = 0.9
    b_bound_ratio: float = 0.2

    _collected: list = field(default_factory=list, init=False, repr=False)
    _cache_ref: Optional[object] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "ZigZagKVPress requires a ScorerPress as the inner press"
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be between 0 and 1"
        assert 0 < self.attention_threshold <= 1, "attention_threshold must be between 0 and 1"
        assert 0 <= self.b_bound_ratio < 1, "b_bound_ratio must be between 0 and 1"

    def post_init_from_model(self, model: PreTrainedModel):
        self.press.post_init_from_model(model)

    def _compute_lmba(self, scores: torch.Tensor) -> float:
        """
        Compute LMBA (Layer Minimum Budget to maintain Attention) for one layer.

        LMBA is the average number of tokens needed to accumulate
        ``attention_threshold`` of the total score mass, measured per head and
        averaged across heads and batch. Concentrated attention (mass in few
        tokens) gives a low LMBA; diffuse attention gives a high LMBA.

        Parameters
        ----------
        scores : torch.Tensor
            Importance scores of shape (batch, num_kv_heads, k_len).
        """
        k_len = scores.shape[-1]
        # Use only non-negative mass for a well-defined cumulative distribution.
        mass = scores.float().clamp(min=0.0)
        sorted_mass, _ = mass.sort(dim=-1, descending=True)
        cumsum = sorted_mass.cumsum(dim=-1)
        total = sorted_mass.sum(dim=-1, keepdim=True)
        threshold = self.attention_threshold * total
        # Number of tokens to reach the threshold (at least 1).
        mba = (cumsum < threshold).sum(dim=-1) + 1
        mba = mba.clamp(max=k_len)
        return mba.float().mean().item()

    def _compute_budgets(self, lmba_values: list[float], seq_len: int) -> list[int]:
        """
        Convert per-layer LMBA values into integer per-layer token budgets.

        Uses the ZigZagKV allocation rule (Equation 6 from the paper):
            uncertainty_l = LMBA_l / sum_k(LMBA_k)
            budget_l = B_bound + (B_avg - B_bound) * L * uncertainty_l

        The total budget sums to L * B_avg, so the average compression ratio is
        preserved.
        """
        num_layers = len(lmba_values)
        total_lmba = sum(lmba_values)

        b_avg = seq_len * (1 - self.compression_ratio)
        b_bound = b_avg * self.b_bound_ratio

        budgets = []
        for lmba in lmba_values:
            if total_lmba == 0:
                budget = b_avg
            else:
                uncertainty = lmba / total_lmba
                budget = b_bound + (b_avg - b_bound) * num_layers * uncertainty
            budget_int = int(round(budget))
            budget_int = max(1, min(budget_int, seq_len))
            budgets.append(budget_int)

        return budgets

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Collection hook: compute and store per-layer scores + LMBA during prefill.

        No compression happens here. Only lightweight state (a score tensor of
        shape (batch, num_kv_heads, k_len) and a scalar LMBA) is retained, which
        keeps memory usage close to a standard (uncompressed) prefill.
        """
        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]

        # Only collect during prefill, not during decoding.
        if kwargs["cache_position"][-1] > q_len:
            return output

        cache = kwargs["past_key_values"]
        if self._cache_ref is None:
            self._cache_ref = cache

        keys, values = extract_keys_and_values(cache, module.layer_idx)
        scores = self.press.score(module, hidden_states, keys, values, output[1], kwargs)
        lmba = self._compute_lmba(scores)

        self._collected.append(
            {
                "layer_idx": module.layer_idx,
                "module": module,
                "scores": scores,
                "lmba": lmba,
            }
        )

        return output

    def _apply_compression(self):
        """
        Deferred compression: compute global budgets, then resize each layer.
        """
        if not self._collected or self.compression_ratio == 0:
            return

        cache = self._cache_ref
        seq_len = self._collected[0]["scores"].shape[-1]
        lmba_values = [c["lmba"] for c in self._collected]
        budgets = self._compute_budgets(lmba_values, seq_len)

        logger.debug(
            f"ZigZagKV per-layer budgets (seq_len={seq_len}): "
            f"min={min(budgets)}, max={max(budgets)}, mean={sum(budgets) / len(budgets):.1f}"
        )

        for data, n_kept in zip(self._collected, budgets):
            module = data["module"]
            layer_idx = data["layer_idx"]
            scores = data["scores"]

            keys, values = extract_keys_and_values(cache, layer_idx)

            if n_kept >= keys.shape[2]:
                continue

            indices = scores.topk(n_kept, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
            keys = keys.gather(2, indices).contiguous()
            values = values.gather(2, indices).contiguous()

            cache_layer = cache.layers[layer_idx]
            if isinstance(cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
                cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
                cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
                cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
                cache_layer.cumulative_length = keys.shape[2]
            else:
                cache_layer.keys = keys
                cache_layer.values = values

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager that applies ZigZagKV compression during prefill.

        Registers hooks that collect per-layer scores and LMBA during the forward
        pass, then compresses every layer with dynamically computed per-layer
        budgets when the context exits.

        Examples
        --------
        >>> from kvpress import ZigZagKVPress, SnapKVPress
        >>> press = ZigZagKVPress(press=SnapKVPress(window_size=32), compression_ratio=0.5)
        >>> with press(model):
        ...     outputs = model(input_ids, past_key_values=cache)
        """
        from transformers import Gemma3ForConditionalGeneration

        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning(f"Model {type(model)} not tested, supported models: {SUPPORTED_MODELS}")

        self._collected = []
        self._cache_ref = None
        self.post_init_from_model(model)

        # Nothing to do without compression; still run a normal forward pass.
        if self.compression_ratio == 0:
            yield
            return

        hooks = []
        try:
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer in language_model.layers:
                if isinstance(model, Gemma3ForConditionalGeneration) and layer.self_attn.is_sliding:
                    continue
                layer.self_attn.rotary_emb = language_model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for hook in hooks:
                hook.remove()

            self._apply_compression()

            self._collected = []
            self._cache_ref = None
