# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from kvpress.attention_patch import _compute_safe_margins, search_hyperplane


def test_search_hyperplane():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bsz, seq_len, head_dim = 50, 500, 128
    X = torch.rand(bsz, seq_len, head_dim, device=device)
    Y = search_hyperplane(X)
    assert torch.exp(torch.bmm(X, Y.unsqueeze(-1))).max() == 0


def test_search_hyperplane_float16_is_finite():
    torch.manual_seed(0)
    X = torch.randn(4, 128, 64, dtype=torch.float16) + 1.0
    attention_scaling = X.shape[-1] ** -0.5

    Y = search_hyperplane(X, attention_scaling=attention_scaling)

    assert Y.dtype == X.dtype
    assert torch.isfinite(Y).all()
    attention_weights = torch.exp(attention_scaling * torch.bmm(X.float(), Y.float().unsqueeze(-1)))
    assert attention_weights.max() == 0


def test_search_hyperplane_random_float16_inputs():
    torch.manual_seed(1)
    bsz, seq_len, head_dim = 4, 256, 128
    attention_scaling = head_dim**-0.5

    for _ in range(50):
        X = torch.randn(bsz, seq_len, head_dim, dtype=torch.float16) + 1.0
        Y = search_hyperplane(X, attention_scaling=attention_scaling)

        assert torch.isfinite(Y).all()
        attention_weights = torch.exp(attention_scaling * torch.bmm(X.float(), Y.float().unsqueeze(-1)))
        assert attention_weights.max() == 0


def test_compute_safe_margins_accounts_for_output_dtype():
    X = torch.tensor([[[2.0, 1.0], [4.0, 2.0]]])
    Y = torch.tensor([[1.0, 1.0]])
    measured_margins = torch.bmm(X, Y.unsqueeze(-1)).squeeze(-1)

    float32_margins = _compute_safe_margins(X, Y, torch.float32)
    float16_margins = _compute_safe_margins(X, Y, torch.float16)

    assert torch.all(float16_margins > 0)
    assert torch.all(float16_margins < float32_margins)
    assert torch.all(float32_margins < measured_margins)


def test_compute_safe_margins_accounts_for_cancellation():
    X = torch.tensor([[[1.0, -0.9995]]])
    Y = torch.tensor([[1.0, 1.0]])
    measured_margin = torch.bmm(X, Y.unsqueeze(-1)).item()

    safe_margin = _compute_safe_margins(X, Y, torch.float16).item()

    assert measured_margin > 0
    assert safe_margin < 0
