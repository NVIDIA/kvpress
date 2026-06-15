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
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class ZigZagKVPress(BasePress):
    """
    Dynamic per-layer KV cache compression based on attention entropy.

    Wrapper press that dynamically allocates per-layer KV cache budgets based on
    layer uncertainty (measured via LMBA — Layer Minimum Budget to maintain Attention).
    Layers with diffuse attention (high entropy) get larger budgets; layers with
    concentrated attention (low entropy) get smaller budgets.

    Based on ZigZagKV (https://arxiv.org/abs/2412.09036, COLING 2025).

    The algorithm works in two phases within a single forward pass:
    1. **Collect phase**: Forward hooks record attention weights and compute LMBA per layer.
    2. **Compress phase**: After the forward pass completes, per-layer budgets are computed
       using the uncertainty-based formula and the inner press compresses each layer.

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method used for token selection within each layer.
    compression_ratio : float, default=0.0
        Average fraction of tokens to remove across all layers.
    window_size : int, default=64
        Number of recent tokens whose attention weights are used to compute LMBA.
    attention_threshold : float, default=0.9
        Fraction of cumulative attention mass used to determine LMBA.
    b_bound_ratio : float, default=0.1
        Minimum budget per layer as a fraction of the average budget size.
        Prevents any layer from being overly compressed.
    """

    press: ScorerPress = field(default_factory=lambda: ScorerPress())
    compression_ratio: float = 0.0
    window_size: int = 64
    attention_threshold: float = 0.9
    b_bound_ratio: float = 0.1

    _lmba_values: list = field(default_factory=list, init=False, repr=False)
    _layer_data: list = field(default_factory=list, init=False, repr=False)
    _cache_ref: Optional[object] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be between 0 and 1"
        assert 0 < self.attention_threshold <= 1, "attention_threshold must be between 0 and 1"
        assert 0 <= self.b_bound_ratio < 1, "b_bound_ratio must be between 0 and 1"
        assert isinstance(self.press, ScorerPress), "ZigZagKVPress requires a ScorerPress as the inner press"

    def _compute_lmba(self, attentions: torch.Tensor, k_len: int) -> float:
        """
        Compute LMBA (Layer Minimum Budget to maintain Attention) for a single layer.

        For each attention head, finds the minimum number of tokens needed to
        capture `attention_threshold` of the total attention mass, using the
        last `window_size` query positions. Returns the average across all heads and batches.
        """
        bsz, num_heads, q_len, _ = attentions.shape

        window = min(self.window_size, q_len)
        attn = attentions[:, :, -window:, :]

        # Average attention over the window queries: (bsz, num_heads, k_len)
        avg_attn = attn.mean(dim=2)

        # Sort descending and compute cumulative sum to find MBA per head
        sorted_attn, _ = avg_attn.sort(dim=-1, descending=True)
        cumsum = sorted_attn.cumsum(dim=-1)

        # MBA = first position where cumulative attention >= threshold
        threshold = self.attention_threshold * avg_attn.sum(dim=-1, keepdim=True)
        mba = (cumsum < threshold).sum(dim=-1) + 1
        mba = mba.clamp(max=k_len)

        return mba.float().mean().item()

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Collection hook: compute LMBA and cache layer data for deferred compression.
        """
        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]

        if kwargs["cache_position"][-1] > q_len:
            return output

        cache = kwargs["past_key_values"]
        if self._cache_ref is None:
            self._cache_ref = cache

        attentions = output[1]
        if attentions is None:
            raise ValueError(
                "ZigZagKVPress requires attention weights. "
                'Use attn_implementation="eager" when loading the model.'
            )

        keys, values = extract_keys_and_values(cache, module.layer_idx)
        k_len = keys.shape[2]

        lmba = self._compute_lmba(attentions, k_len)
        self._lmba_values.append(lmba)

        self._layer_data.append({
            "layer_idx": module.layer_idx,
            "module": module,
            "hidden_states": hidden_states,
            "keys": keys,
            "values": values,
            "attentions": attentions,
            "kwargs": kwargs,
        })

        return output

    def _compute_compression_ratios(self, seq_len: int) -> list[float]:
        """
        Convert LMBA values to per-layer compression ratios.

        Uses the ZigZagKV budget allocation formula (Equation 6 from the paper):
            uncertainty_l = LMBA_l / sum(LMBA)
            budget_l = B_bound + (B_avg - B_bound) * L * uncertainty_l
            compression_ratio_l = 1 - budget_l / seq_len
        """
        num_layers = len(self._lmba_values)
        total_lmba = sum(self._lmba_values)

        if total_lmba == 0:
            return [self.compression_ratio] * num_layers

        b_avg = seq_len * (1 - self.compression_ratio)
        b_bound = b_avg * self.b_bound_ratio

        ratios = []
        for lmba in self._lmba_values:
            uncertainty = lmba / total_lmba
            budget = b_bound + (b_avg - b_bound) * num_layers * uncertainty
            budget = max(1.0, min(budget, float(seq_len)))
            ratio = 1.0 - budget / seq_len
            ratio = max(0.0, min(ratio, 1.0 - 1.0 / seq_len))
            ratios.append(ratio)

        return ratios

    def _apply_compression(self):
        """
        Phase 2: compress all layers using dynamically computed per-layer ratios.

        All layers are padded to the same final length (the maximum across layers)
        to ensure compatibility with HuggingFace's attention mask mechanism, which
        expects uniform cache sizes across layers.
        """
        if not self._layer_data:
            return

        cache = self._cache_ref
        seq_len = self._layer_data[0]["keys"].shape[2]
        compression_ratios = self._compute_compression_ratios(seq_len)

        logger.debug(
            f"ZigZagKV per-layer compression ratios: "
            f"min={min(compression_ratios):.3f}, max={max(compression_ratios):.3f}, "
            f"mean={sum(compression_ratios) / len(compression_ratios):.3f}"
        )

        compressed_layers = []
        for data, ratio in zip(self._layer_data, compression_ratios):
            module = data["module"]

            original_ratio = self.press.compression_ratio
            self.press.compression_ratio = ratio

            keys, values = self.press.compress(
                module,
                data["hidden_states"],
                data["keys"],
                data["values"],
                data["attentions"],
                data["kwargs"],
            )

            self.press.compression_ratio = original_ratio
            compressed_layers.append((data["layer_idx"], keys, values))

        max_len = max(k.shape[2] for _, k, _ in compressed_layers)

        for layer_idx, keys, values in compressed_layers:
            pad_len = max_len - keys.shape[2]
            if pad_len > 0:
                keys = torch.cat([
                    keys,
                    torch.zeros(*keys.shape[:2], pad_len, keys.shape[3], dtype=keys.dtype, device=keys.device),
                ], dim=2)
                values = torch.cat([
                    values,
                    torch.zeros(*values.shape[:2], pad_len, values.shape[3], dtype=values.dtype, device=values.device),
                ], dim=2)

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
        Context manager that applies ZigZagKV compression.

        Registers hooks to collect attention patterns during prefill, then
        compresses all layers with dynamically computed per-layer budgets
        when the context manager exits.

        Parameters
        ----------
        model : PreTrainedModel
            The transformer model to apply compression to.

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

        self._lmba_values = []
        self._layer_data = []
        self._cache_ref = None
        self.press.post_init_from_model(model)
        self.post_init_from_model(model)

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

            self._lmba_values = []
            self._layer_data = []
            self._cache_ref = None
