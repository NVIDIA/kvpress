# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Triton kernels for EchoPress virtual scoring.

The first pass computes log-sum-exp over sink, cached, and causal-copy keys.
The second recomputes cached-key logits and takes the maximum probability per key.
"""
import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _lse_kernel(
        Q1,
        Q2,
        K,
        KS,
        LSE,
        m,
        sink,
        scale,
        stride_qh,
        stride_qg,
        stride_qm,
        stride_kh,
        stride_km,
        stride_ksh,
        stride_ksm,
        stride_lh,
        stride_lg,
        HAS_SINK: tl.constexpr,
        BQ: tl.constexpr,
        BK: tl.constexpr,
        BS: tl.constexpr,
        D: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        h = tl.program_id(1)
        g = tl.program_id(2)
        offs_q = pid_q * BQ + tl.arange(0, BQ)
        offs_d = tl.arange(0, D)
        q_mask = offs_q < m
        q_ptrs = h * stride_qh + g * stride_qg + offs_q[:, None] * stride_qm + offs_d[None, :]
        q1 = tl.load(Q1 + q_ptrs, mask=q_mask[:, None], other=0.0)
        q2 = tl.load(Q2 + q_ptrs, mask=q_mask[:, None], other=0.0)
        m_i = tl.full([BQ], float("-inf"), tl.float32)
        l_i = tl.zeros([BQ], tl.float32)
        if HAS_SINK:
            offs_s = tl.arange(0, BS)
            s_mask = offs_s < sink
            ks = tl.load(
                KS + h * stride_ksh + offs_s[:, None] * stride_ksm + offs_d[None, :], mask=s_mask[:, None], other=0.0
            )
            s = tl.dot(q2, tl.trans(ks)) * scale
            s = tl.where(s_mask[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(s - m_new[:, None]), 1)
            m_i = m_new
        for k0 in range(0, m, BK):
            offs_k = k0 + tl.arange(0, BK)
            k_mask = offs_k < m
            k = tl.load(
                K + h * stride_kh + offs_k[:, None] * stride_km + offs_d[None, :], mask=k_mask[:, None], other=0.0
            )
            kt = tl.trans(k)
            s2 = tl.dot(q2, kt) * scale
            s2 = tl.where(k_mask[None, :], s2, float("-inf"))
            s1 = tl.dot(q1, kt) * scale
            s1 = tl.where(k_mask[None, :] & (offs_k[None, :] <= offs_q[:, None]), s1, float("-inf"))
            m_new = tl.maximum(m_i, tl.maximum(tl.max(s2, 1), tl.max(s1, 1)))
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(tl.exp(s2 - m_new[:, None]), 1) + tl.sum(tl.exp(s1 - m_new[:, None]), 1)
            m_i = m_new
        lse = m_i + tl.log(l_i)
        tl.store(LSE + h * stride_lh + g * stride_lg + offs_q, lse, mask=q_mask)

    @triton.jit
    def _colmax_kernel(
        Q2,
        K,
        LSE,
        OUT,
        m,
        scale,
        stride_qh,
        stride_qg,
        stride_qm,
        stride_kh,
        stride_km,
        stride_lh,
        stride_lg,
        stride_oh,
        BQ: tl.constexpr,
        BK: tl.constexpr,
        D: tl.constexpr,
    ):
        pid_k = tl.program_id(0)
        h = tl.program_id(1)
        g = tl.program_id(2)
        offs_k = pid_k * BK + tl.arange(0, BK)
        offs_d = tl.arange(0, D)
        k_mask = offs_k < m
        k = tl.load(K + h * stride_kh + offs_k[:, None] * stride_km + offs_d[None, :], mask=k_mask[:, None], other=0.0)
        kt = tl.trans(k)
        acc = tl.full([BK], float("-inf"), tl.float32)
        for q0 in range(0, m, BQ):
            offs_q = q0 + tl.arange(0, BQ)
            q_mask = offs_q < m
            q2 = tl.load(
                Q2 + h * stride_qh + g * stride_qg + offs_q[:, None] * stride_qm + offs_d[None, :],
                mask=q_mask[:, None],
                other=0.0,
            )
            lse = tl.load(LSE + h * stride_lh + g * stride_lg + offs_q, mask=q_mask, other=float("inf"))
            v = tl.dot(q2, kt) * scale - lse[:, None]
            v = tl.where(q_mask[:, None], v, float("-inf"))
            acc = tl.maximum(acc, tl.max(v, 0))
        tl.atomic_max(OUT + h * stride_oh + offs_k, acc, mask=k_mask)


def virtual_scores_triton(
    q1: torch.Tensor,
    q2: torch.Tensor,
    k_chunk: torch.Tensor,
    k_sink: torch.Tensor,
    block_q: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Return fp32 virtual scores with shape `[Hkv, m]`."""
    assert HAS_TRITON, "triton is not available"
    Hkv, G, m, d = q1.shape
    sink = k_sink.shape[1]
    assert d in (64, 128, 256), "head_dim must be 64, 128 or 256"
    assert q1.stride(-1) == 1 and q2.stride(-1) == 1 and k_chunk.stride(-1) == 1
    scale = 1.0 / math.sqrt(d)
    lse = torch.empty(Hkv, G, m, dtype=torch.float32, device=q1.device)
    out = torch.full((Hkv, m), float("-inf"), dtype=torch.float32, device=q1.device)
    bs = max(16, triton.next_power_of_2(sink)) if sink > 0 else 16
    grid = (triton.cdiv(m, block_q), Hkv, G)
    _lse_kernel[grid](
        q1,
        q2,
        k_chunk,
        k_sink if sink > 0 else k_chunk,
        lse,
        m,
        sink,
        scale,
        q1.stride(0),
        q1.stride(1),
        q1.stride(2),
        k_chunk.stride(0),
        k_chunk.stride(1),
        k_sink.stride(0) if sink > 0 else 0,
        k_sink.stride(1) if sink > 0 else 0,
        lse.stride(0),
        lse.stride(1),
        HAS_SINK=sink > 0,
        BQ=block_q,
        BK=block_k,
        BS=bs,
        D=d,
        num_warps=4,
    )
    grid = (triton.cdiv(m, block_k), Hkv, G)
    _colmax_kernel[grid](
        q2,
        k_chunk,
        lse,
        out,
        m,
        scale,
        q2.stride(0),
        q2.stride(1),
        q2.stride(2),
        k_chunk.stride(0),
        k_chunk.stride(1),
        lse.stride(0),
        lse.stride(1),
        out.stride(0),
        BQ=block_q,
        BK=block_k,
        D=d,
        num_warps=4,
    )
    return out.exp()
