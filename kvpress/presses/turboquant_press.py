# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn
from transformers import QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.utils import extract_keys_and_values

TurboQuantObjective = Literal["mse", "prod"]


@dataclass
class TurboQuantPress(BasePress):
    """
    TurboQuant KV cache quantization.

    This implements the algorithms from "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
    (https://arxiv.org/abs/2504.19874):
    - TurboQuant_mse: random rotation, Lloyd-Max scalar codebook, nearest-centroid quantization, inverse rotation.
    - TurboQuant_prod: TurboQuant_mse with one fewer bit plus a 1-bit QJL residual correction.

    The current BasePress interface stores the returned cache tensors in the model cache, so this press applies the
    paper's quantize/dequantize transform to the KV tensors. It does not bit-pack the cache representation.

    Parameters
    ----------
    bit_width : int, default=3
        Bits per channel. For the `prod` objective, one bit is reserved for the QJL residual.
    key_objective : {"mse", "prod"}, default="prod"
        Quantizer used for keys. Keys are involved in query-key inner products, so `prod` follows the paper's
        unbiased inner-product quantizer.
    value_objective : {"mse", "prod"}, default="mse"
        Quantizer used for values. Values are reconstructed vectors, so `mse` is the default.
    seed : int, default=0
        Seed for the paper's random rotation and QJL projection matrices.
    codebook_grid_size : int, default=8193
        Number of grid points used to numerically solve the one-dimensional continuous k-means problem.
    max_lloyd_iterations : int, default=100
        Maximum Lloyd-Max refinement iterations for the scalar codebook.
    lloyd_tolerance : float, default=1e-6
        Stop codebook refinement when all centroids move by less than this value.
    full_precision_bits : int, default=16
        Reference precision used only to expose an approximate `compression_ratio` attribute.
    """

    bit_width: int = 3
    key_objective: TurboQuantObjective = "prod"
    value_objective: TurboQuantObjective = "mse"
    seed: int = 0
    codebook_grid_size: int = 8193
    max_lloyd_iterations: int = 100
    lloyd_tolerance: float = 1e-6
    full_precision_bits: int = 16
    supports_decoding: bool = True
    persistent_across_phases: bool = True
    compression_ratio: float = field(init=False)
    _codebook_cache: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _rotation_cache: dict[tuple[int, str, int], torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _projection_cache: dict[tuple[int, str, int], torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        assert isinstance(self.bit_width, int), "bit_width must be an integer"
        assert 1 <= self.bit_width <= 8, "bit_width must be between 1 and 8"
        assert self.key_objective in ("mse", "prod"), "key_objective must be 'mse' or 'prod'"
        assert self.value_objective in ("mse", "prod"), "value_objective must be 'mse' or 'prod'"
        assert self.codebook_grid_size >= 257, "codebook_grid_size must be at least 257"
        assert self.max_lloyd_iterations > 0, "max_lloyd_iterations must be positive"
        assert self.lloyd_tolerance > 0, "lloyd_tolerance must be positive"
        assert self.full_precision_bits > 0, "full_precision_bits must be positive"
        self.compression_ratio = max(0.0, 1.0 - self.bit_width / self.full_precision_bits)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del module, hidden_states, attentions, kwargs
        return self._quantize_tensor(keys, self.key_objective), self._quantize_tensor(values, self.value_objective)

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        del input
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        cache_layer = cache.layers[module.layer_idx]
        q_len = hidden_states.shape[1]

        keys, values = extract_keys_and_values(cache, module.layer_idx)
        if kwargs["cache_position"][-1] > q_len:
            key_prefix, key_suffix = keys[:, :, :-q_len, :], keys[:, :, -q_len:, :]
            value_prefix, value_suffix = values[:, :, :-q_len, :], values[:, :, -q_len:, :]
            keys = torch.cat([key_prefix, self._quantize_tensor(key_suffix, self.key_objective)], dim=2)
            values = torch.cat([value_prefix, self._quantize_tensor(value_suffix, self.value_objective)], dim=2)
        else:
            keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)

        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys = keys
            cache_layer.values = values

        return output

    def _quantize_tensor(self, tensor: torch.Tensor, objective: TurboQuantObjective) -> torch.Tensor:
        shape = tensor.shape
        vectors = tensor.reshape(-1, shape[-1])
        quantized = self._quantize_vectors(vectors, objective)
        return quantized.reshape(shape).to(dtype=tensor.dtype)

    def _quantize_vectors(self, vectors: torch.Tensor, objective: TurboQuantObjective) -> torch.Tensor:
        dtype = vectors.dtype
        vectors = vectors.to(dtype=torch.float32)
        norms = vectors.norm(dim=-1, keepdim=True)
        unit_vectors = vectors / norms.clamp_min(torch.finfo(vectors.dtype).tiny)

        if objective == "mse":
            reconstructed = self._turboquant_mse(unit_vectors, self.bit_width)
        else:
            reconstructed = self._turboquant_prod(unit_vectors, self.bit_width)

        return (reconstructed * norms).to(dtype=dtype)

    def _turboquant_mse(self, unit_vectors: torch.Tensor, bit_width: int) -> torch.Tensor:
        dimension = unit_vectors.shape[-1]
        if bit_width == 0:
            return torch.zeros_like(unit_vectors)

        rotation = self._get_rotation(dimension, unit_vectors.device)
        codebook = self._get_codebook(dimension, bit_width, unit_vectors.device)
        boundaries = (codebook[:-1] + codebook[1:]) / 2

        rotated = unit_vectors @ rotation.T
        indices = torch.bucketize(rotated.contiguous(), boundaries)
        quantized_rotated = codebook[indices]
        return quantized_rotated @ rotation

    def _turboquant_prod(self, unit_vectors: torch.Tensor, bit_width: int) -> torch.Tensor:
        dimension = unit_vectors.shape[-1]
        mse_reconstruction = self._turboquant_mse(unit_vectors, bit_width - 1)
        residual = unit_vectors - mse_reconstruction
        gamma = residual.norm(dim=-1, keepdim=True)
        projection = self._get_projection(dimension, unit_vectors.device)

        qjl = torch.where(
            residual @ projection.T >= 0,
            torch.ones((), dtype=unit_vectors.dtype, device=unit_vectors.device),
            -torch.ones((), dtype=unit_vectors.dtype, device=unit_vectors.device),
        )
        qjl_reconstruction = math.sqrt(math.pi / 2) / dimension * gamma * (qjl @ projection)
        return mse_reconstruction + qjl_reconstruction

    def _get_rotation(self, dimension: int, device: torch.device) -> torch.Tensor:
        key = (dimension, device.type, device.index or -1)
        if key not in self._rotation_cache:
            generator = torch.Generator(device="cpu").manual_seed(self.seed + 104729 * dimension)
            gaussian = torch.randn(dimension, dimension, generator=generator, dtype=torch.float32)
            q, r = torch.linalg.qr(gaussian)
            signs = torch.sign(torch.diagonal(r))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            self._rotation_cache[key] = (q * signs.unsqueeze(0)).to(device=device)
        return self._rotation_cache[key]

    def _get_projection(self, dimension: int, device: torch.device) -> torch.Tensor:
        key = (dimension, device.type, device.index or -1)
        if key not in self._projection_cache:
            generator = torch.Generator(device="cpu").manual_seed(self.seed + 1299709 * dimension)
            projection = torch.randn(
                dimension,
                dimension,
                generator=generator,
                dtype=torch.float32,
            )
            self._projection_cache[key] = projection.to(device=device)
        return self._projection_cache[key]

    def _get_codebook(self, dimension: int, bit_width: int, device: torch.device) -> torch.Tensor:
        key = (dimension, bit_width)
        if key not in self._codebook_cache:
            self._codebook_cache[key] = self._build_codebook(dimension, bit_width)
        return self._codebook_cache[key].to(device=device)

    def _build_codebook(self, dimension: int, bit_width: int) -> torch.Tensor:
        if bit_width == 0:
            return torch.zeros(1, dtype=torch.float32)

        n_centroids = 2**bit_width
        grid_size = max(self.codebook_grid_size, 64 * n_centroids + 1)
        if grid_size % 2 == 0:
            grid_size += 1

        grid = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float64)
        density_base = torch.clamp(1 - grid.square(), min=torch.finfo(torch.float64).tiny)
        log_weights = (dimension - 3) / 2 * torch.log(density_base)
        weights = torch.exp(log_weights - log_weights.max())

        span = min(1.0, 3.0 / math.sqrt(dimension))
        centroids = torch.linspace(-span, span, n_centroids, dtype=torch.float64)
        for _ in range(self.max_lloyd_iterations):
            boundaries = (centroids[:-1] + centroids[1:]) / 2
            assignments = torch.bucketize(grid, boundaries)
            updated = centroids.clone()
            for idx in range(n_centroids):
                mask = assignments == idx
                bucket_weight = weights[mask].sum()
                if bucket_weight > 0:
                    updated[idx] = (grid[mask] * weights[mask]).sum() / bucket_weight

            if torch.max(torch.abs(updated - centroids)) < self.lloyd_tolerance:
                centroids = updated
                break
            centroids = updated

        return centroids.to(dtype=torch.float32)
