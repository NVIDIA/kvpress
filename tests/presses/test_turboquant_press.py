# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from kvpress import TurboQuantPress


def test_turboquant_preserves_shape_dtype_and_zero_vectors():
    press = TurboQuantPress(bit_width=2, seed=7, codebook_grid_size=513, max_lloyd_iterations=20)
    keys = torch.randn(2, 3, 5, 16, dtype=torch.float32)
    zeros = torch.zeros_like(keys)

    quantized = press._quantize_tensor(keys, "prod")
    quantized_zeros = press._quantize_tensor(zeros, "prod")

    assert quantized.shape == keys.shape
    assert quantized.dtype == keys.dtype
    assert torch.allclose(quantized_zeros, zeros)


def test_turboquant_mse_error_decreases_with_bit_width():
    generator = torch.Generator().manual_seed(0)
    vectors = torch.randn(128, 16, generator=generator)
    low_bit = TurboQuantPress(bit_width=1, seed=11, codebook_grid_size=513, max_lloyd_iterations=20)
    high_bit = TurboQuantPress(bit_width=4, seed=11, codebook_grid_size=513, max_lloyd_iterations=20)

    low_error = torch.mean((low_bit._quantize_vectors(vectors, "mse") - vectors) ** 2)
    high_error = torch.mean((high_bit._quantize_vectors(vectors, "mse") - vectors) ** 2)

    assert high_error < low_error
