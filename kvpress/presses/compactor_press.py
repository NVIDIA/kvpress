# SPDX-FileCopyrightText: Copyright Vivek Chari
# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
from transformers.models.phi3.modeling_phi3 import Phi3Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.utils import get_query_states


@dataclass
class CompactorPress(ScorerPress):
    """
    Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores

    Compactor blends: geometry-based outlier scores via (approximate) statistical leverage on key
    embeddings; and non-causal, chunked attention. Currently only supports prefill. The presented
    version slightly differs from the paper in that: (1) we set blending=compression_ratio by default,
    which is a good heuristic and should work for most users and (2) we use a cholesky
    decomposition to compute the leverage scores. Please see the paper for an in-depth discussion.

    References:
    - Chari & Van Durme (2025): "Compactor: Calibrated Query-Agnostic KV Cache
      Compression with Approximate Leverage Scores" (https://arxiv.org/pdf/2507.08143v1)

    Parameters
    ----------
    compression_ratio : float, default ``0.0``
         Fraction of key-value pairs to remove during compression.
    sink_size_start : int, default ``8``
        Number of initial sink tokens to always protect.
    sink_size_end : int, default ``4``
        Number of most-recent tokens to always protect.
    chunk_size : int, default ``256``
        Chunk size used to in non-causal attention.
    blending : Optional[float], default ``None``
        Weight for blending scores in the final output. If ``None``,
        it set to ``compression_ratio``, which tends to be a good default.

    Output
    ------
    score(...) returns a tensor of shape (B, H_kv, S) with higher values
    indicating more important tokens for retention.
    """

    compression_ratio: float = 0.0
    sink_size_start: int = 8
    sink_size_end: int = 4
    chunk_size: int = 256
    blending: Optional[float] = None

    @staticmethod
    def compute_chunked_attention(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        chunk_size: int,
        position_embeddings: tuple,
    ) -> torch.Tensor:
        """Non-causal, chunked attention scores"""
        q_len, num_key_value_groups = query_states.shape[-2], query_states.shape[1] // value_states.shape[1]
        # apply RoPE
        cos, sin = position_embeddings
        query_states = (query_states * cos[:, -q_len:, :].unsqueeze(1)) + (
            rotate_half(query_states) * sin[:, -q_len:, :].unsqueeze(1)
        )
        A = CompactorPress.non_causal_chunked_attn(
            query_states,
            repeat_kv(key_states, num_key_value_groups),
            chunk_size,
        )
        # average across query-head groups back to H_kv
        A = A.view(A.shape[0], value_states.shape[1], -1, A.shape[-1]).mean(dim=-2)

        scores = A * value_states.norm(dim=-1)
        scores = F.avg_pool1d(scores, kernel_size=3, padding=3 // 2, stride=1)
        return (scores - scores.mean(dim=-1, keepdim=True)) / scores.std(dim=-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def compute_leverage_scores(
        key_states: torch.Tensor,
        sketch_dimension: int,
    ) -> torch.Tensor:
        """Approximate leverage scores on pre-RoPE keys via right Gaussian sketching."""
        d, k = key_states.shape[-1], sketch_dimension

        # right Gaussian sketch, see paper for theoritcal analysis of this *right* sketch.
        R = torch.randn(
            key_states.shape[0],
            key_states.shape[1],
            d,
            k,
            device=key_states.device,
            dtype=key_states.dtype,
        ) * (1 / math.sqrt(k))

        # sequence-centering then sketch
        key_states = key_states - key_states.mean(dim=-2, keepdim=True)
        key_states = torch.matmul(key_states, R).to(torch.float32)
        gram_matrix = key_states.transpose(-2, -1) @ key_states
        L = CompactorPress.chol_with_jitter(
            0.5 * (gram_matrix + gram_matrix.transpose(-2, -1)), jitter=1e-2, max_tries=5
        )
        # X = (A^T A + \lambda I)^{-1} A^T
        X = torch.cholesky_solve(key_states.transpose(-2, -1), L, upper=False)
        # Y = X^T = A (A^T A + \lambda I)^{-1}
        scores = (key_states * X.transpose(-2, -1)).sum(dim=-1).clamp_min(0)  # ridge-regularized
        return (scores - scores.mean(dim=-1, keepdim=True)) / scores.std(dim=-1, keepdim=True).clamp_min(1e-6)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """Blend leverage and non-causal attention into final importance scores"""
        n_queries = hidden_states.shape[-2]
        assert keys.shape[-2] == n_queries, "CompactorPress only supports prefill at the moment"
        left_keep = min(self.sink_size_start, n_queries)
        right_keep = min(self.sink_size_end, max(0, n_queries - left_keep))
        start_idx, end_idx = left_keep, (None if right_keep == 0 else -right_keep)

        pre_rope_key_states = CompactorPress.get_prerope_key_states(module, hidden_states[:, start_idx:end_idx])
        query_states = get_query_states(module, hidden_states[:, start_idx:end_idx])

        l_scores = self.compute_leverage_scores(
            key_states=pre_rope_key_states,
            sketch_dimension=48,
        )

        cos, sin = kwargs["position_embeddings"]
        attn_scores = self.compute_chunked_attention(
            query_states=query_states,
            key_states=keys[..., start_idx:end_idx, :],
            value_states=values[..., start_idx:end_idx, :],
            chunk_size=self.chunk_size,
            position_embeddings=(cos[..., start_idx:end_idx, :], sin[..., start_idx:end_idx, :]),
        )
        # sanity check. this breaks when not in prefill
        assert attn_scores.shape == l_scores.shape, "CompactorPress only supports prefill at the moment"
        blending = self.blending if self.blending is not None else self.compression_ratio
        scores = blending * l_scores + attn_scores
        # protect sinks by padding
        scores = F.pad(scores, (left_keep, right_keep), value=scores.detach().max())
        return scores

    @staticmethod
    def non_causal_chunked_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """Compute non-causal, chunked attention column-sums over the sequence.
        The sequence is left/right padded to a multiple of ``chunk_size`` and then
        processed in fixed-size tiles

        Parameters
        ----------
        q, k : torch.Tensor, shape (B, H, S, d)
            Query/Key tensors for a single layer/head group.
        chunk_size : int
            Size of the chunk used to tile the sequence axis.
        Returns
        -------
        torch.Tensor, shape (B, H, S)
            Column-wise non-causal attention accumulations per key position.
        """
        # if we are decoding, q is assumed to be reasonably small compared to k, so we skip the chunking all together.
        assert chunk_size > 0, "chunk_size must be positive"
        assert q.shape[-2] == k.shape[-2], "only used in prefill"
        B, H, S, d = k.shape
        # pad to a multiple of chunk_size for easy reshaping
        S_pad = math.ceil(S / chunk_size) * chunk_size
        pad_len = S_pad - S
        if pad_len > 0:
            q_padded = torch.cat([q, torch.zeros(B, H, pad_len, d, device=q.device, dtype=q.dtype)], dim=2)
            k_padded = torch.cat([k, torch.zeros(B, H, pad_len, d, device=k.device, dtype=k.dtype)], dim=2)
        else:
            q_padded, k_padded = q, k

        valid = torch.arange(S_pad, device=q.device) < S
        valid = valid.view(1, 1, S_pad).expand(B, H, S_pad)

        # reshape masks to chunked layout
        num_chunks = S_pad // chunk_size
        query_mask = valid.view(B, H, num_chunks, chunk_size)
        key_mask = valid.view(B, H, num_chunks, chunk_size)

        attn_sum = CompactorPress.chunked_attn_inner(q_padded, k_padded, query_mask, key_mask, chunk_size)
        return attn_sum[..., :S]

    @staticmethod
    def chunked_attn_inner(
        q: torch.Tensor,
        k: torch.Tensor,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """
        helper method for non_causal_chunked_attn
        """
        B, H, S_pad, d = q.shape
        num_chunks = S_pad // chunk_size

        # (B, H, num_chunks, chunk_size, d)
        q_chunks = q.view(B, H, num_chunks, chunk_size, d)
        k_chunks = k.view(B, H, num_chunks, chunk_size, d)

        # (B, H, num_chunks, chunk_size, chunk_size)
        dots = torch.matmul(q_chunks, k_chunks.transpose(-2, -1))
        mask_value = torch.finfo(q.dtype).min
        dots = dots.masked_fill(~query_mask.unsqueeze(-1), mask_value)
        dots = dots.masked_fill(~key_mask.unsqueeze(-2), mask_value)

        attn = torch.softmax(dots.to(torch.float32), dim=-1)
        # sum over query dimension
        return attn.sum(dim=-2).view(B, H, S_pad)

    @staticmethod
    def chol_with_jitter(G: torch.Tensor, jitter: float = 0.0, max_tries: int = 5):
        """
        cholesky factorization with adaptive jitter.
        Args:
            G: Tensor of shape (..., n, n), symmetric
            max_tries: maximum # of jitter escalation attempts.
        Returns:
            L: Tensor that holds the lower-triangular Cholesky factor
        Raises:
            RuntimeError: If a valid Cholesky factor is not found within `max_tries`.
        """
        identity = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
        cur = float(jitter)
        for attempt in range(max_tries):
            L, info = torch.linalg.cholesky_ex(G + cur * identity, upper=False)
            if bool((info == 0).all()):
                return L
            cur = max(1e-8, (1e-2 if cur == 0.0 else 10.0 * cur))
        raise RuntimeError(f"Cholesky failed after {max_tries} tries.")

    @staticmethod
    def get_prerope_key_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Extracts the key states from a given attention module and hidden states tensor.

        This function supports multiple attention module types: Phi3Attention, Qwen3Attention, Gemma3Attention,
        and Llama-like modules. It handles the appropriate projection and reshaping to obtain the key states
        in the expected format.

        Parameters
        ----------
        module : nn.Module
            The attention module from which to extract key states. Must be one of
            Phi3Attention, Qwen3Attention, Gemma3Attention, or a Llama-like attention module
            with a 'k_proj' attribute.
        hidden_states : torch.Tensor
            The input hidden states of shape (batch_size, seq_len, hidden_dim).

        Returns
        -------
        key_states : torch.Tensor
            The extracted key states of shape (batch_size, num_heads, seq_len, head_dim).
        """
        bsz, k_len, _ = hidden_states.shape
        head_dim = module.head_dim
        if isinstance(module, Phi3Attention):
            qkv = module.qkv_proj(hidden_states)
            query_pos = module.config.num_attention_heads * module.head_dim
            key_states = qkv[..., query_pos : query_pos + module.num_key_value_heads * module.head_dim]
        elif hasattr(module, "k_proj"):
            # Assume Llama-like attention layer
            key_states = module.k_proj(hidden_states)
        else:
            raise NotImplementedError(f"Press not yet implemented for {module.__class__}.")

        key_states = key_states.view(bsz, k_len, -1, head_dim).transpose(1, 2)

        # Support for Qwen3 and Gemma3 QK norm
        if isinstance(module, (Qwen3Attention, Gemma3Attention)):
            key_states = module.k_norm(key_states)
        return key_states
