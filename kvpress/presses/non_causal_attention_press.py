# SPDX-FileCopyrightText: Copyright Vivek Chari
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_query_states


@dataclass
class NonCausalAttnPress(ScorerPress):
    """
    Non-causal, chunked attention scorer.

    This press implements the non-causal, chunked attention, sum-over-queries scoring
    used in Compactor. Scores are z-normalized.

    References:
    - Chari & Van Durme (2025): "Compactor: Calibrated Query-Agnostic KV Cache
      Compression with Approximate Leverage Scores" (https://arxiv.org/pdf/2507.08143v1)


    Parameters
    ----------
    chunk_size : int, default ``256``
        Chunk size used in non-causal attention.

    Output
    ------
    score(...) returns a tensor of shape (B, H_kv, S) with higher values
    indicating more important tokens for retention.

    Notes
    -----
    Only supports prefill.
    """

    chunk_size: int = 256

    @staticmethod
    def non_causal_chunked_attn(q: torch.Tensor, k: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Compute non-causal, chunked attention column-sums over the sequence.
        The sequence is left/right padded to a multiple of ``chunk_size`` and then
        processed in fixed-size tiles.

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
        assert chunk_size > 0, "chunk_size must be positive"
        assert q.shape[-2] == k.shape[-2], "only used in prefill"
        B, H, S, d = k.shape
        # pad to a multiple of chunk_size for easy reshaping
        S_pad = math.ceil(S / chunk_size) * chunk_size
        pad_len = S_pad - S

        if pad_len > 0:
            q_padded = torch.cat([q, torch.zeros(B, H, pad_len, d, device=q.device, dtype=q.dtype)], dim=2)
            k_padded = torch.cat([k, torch.zeros(B, H, pad_len, d, device=k.device, dtype=k.dtype)], dim=2)
            last_chunk_start = (S // chunk_size) * chunk_size
            in_valid = torch.arange(last_chunk_start, S_pad, device=q.device) >= S
            query_mask = key_mask = in_valid.view(1, 1, chunk_size).expand(B, H, chunk_size)
        else:
            q_padded, k_padded = q, k
            last_chunk_start = ((S - 1) // chunk_size) * chunk_size
            in_valid = torch.arange(last_chunk_start, S_pad, device=q.device) >= S
            query_mask = key_mask = in_valid.view(1, 1, chunk_size).expand(B, H, chunk_size)

        num_chunks = S_pad // chunk_size
        # (B, H, num_chunks, chunk_size, d)
        q_chunks = q_padded.view(B, H, num_chunks, chunk_size, d)
        k_chunks = k_padded.view(B, H, num_chunks, chunk_size, d)

        # (B, H, num_chunks, chunk_size, chunk_size)
        dots = torch.matmul(q_chunks, k_chunks.transpose(-2, -1))
        dots[:, :, -1].masked_fill_(query_mask.unsqueeze(-1), 0)
        dots[:, :, -1].masked_fill_(key_mask.unsqueeze(-2), -1e-9)
        attn = torch.softmax(dots.to(torch.float32), dim=-1)
        # sum over query and trim padding
        return attn.sum(dim=-2).view(B, H, S_pad)[..., :S]

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        n_queries = hidden_states.shape[-2]
        assert keys.shape[-2] == n_queries, "NonCausalAttnPress only supports prefill"

        cos, sin = kwargs["position_embeddings"]
        q = get_prerope_query_states(module, hidden_states)  # (B, H_q, S, d)

        q_len = q.shape[-2]
        num_kv_groups = q.shape[1] // values.shape[1]
        # apply RoPE to the queries for the last q_len positions
        q = (q * cos[:, -q_len:, :].unsqueeze(1)) + (rotate_half(q) * sin[:, -q_len:, :].unsqueeze(1))

        A = self.non_causal_chunked_attn(q, repeat_kv(keys, num_kv_groups), self.chunk_size)  # (B, H_q, S)
        # average across query-head groups back to H_kv
        A = A.view(A.shape[0], values.shape[1], -1, A.shape[-1]).mean(dim=-2)  # (B, H_kv, S)

        scores = A * values.norm(dim=-1)  # (B, H_kv, S)
        scores = F.avg_pool1d(scores, kernel_size=3, padding=1, stride=1)
        z_scores = (scores - scores.mean()) / scores.std().clamp_min(1e-6)  # head-wise z-norm
        return z_scores
