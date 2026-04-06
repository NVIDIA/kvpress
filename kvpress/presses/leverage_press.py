# SPDX-FileCopyrightText: Copyright Vivek Chari
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_key_states


@dataclass
class LeverageScorePress(ScorerPress):
    """
    Approximate leverage-score scorer on pre-RoPE keys.

    Computes geometry-based outlier scores via (approximate) statistical leverage
    on key embeddings using a right Gaussian sketch. Scores are z-score normalized.
    The presented version slightly differs from the paper in that: we use a cholesky
    decomposition to compute the leverage scores. Please see the paper for an in-depth
    discussion.

    References:
    - Chari & Van Durme (2025): "Compactor: Calibrated Query-Agnostic KV Cache
      Compression with Approximate Leverage Scores" (https://arxiv.org/pdf/2507.08143v1)

    Parameters
    ----------
    sketch_dimension : int, default ``48``
        Size of Gaussian sketch.

    Output
    ------
    score(...) returns a tensor of shape (B, H_kv, S) with higher values
    indicating more important tokens for retention.

    Notes
    -----
    Currently only supports prefill.
    """

    sketch_dimension: int = 48

    @staticmethod
    def chol_with_jitter(G: torch.Tensor, jitter: float = 0.0, max_tries: int = 5):
        """cholesky factorization with adaptive jitter."""
        identity = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
        cur = float(jitter)
        for _ in range(max_tries):
            L, info = torch.linalg.cholesky_ex(G + cur * identity, upper=False)
            if bool((info == 0).all()):
                return L
            cur = max(1e-8, (1e-2 if cur == 0.0 else 10.0 * cur))
        raise RuntimeError(f"Cholesky failed after {max_tries} tries.")

    @staticmethod
    def compute_leverage_scores(key_states: torch.Tensor, sketch_dimension: int) -> torch.Tensor:
        """
        Approximate leverage scores on pre-RoPE keys via right Gaussian sketching. We
        use a Cholesky solve to do this efficiently.
        """
        d, k = key_states.shape[-1], sketch_dimension
        # right Gaussian sketch, see paper for theoritcal analysis of this *right* sketch.
        #
        Phi = torch.randn(
            key_states.shape[0],
            key_states.shape[1],
            d,
            k,
            device=key_states.device,
            dtype=key_states.dtype,
        ) * (1 / math.sqrt(k))

        # sequence-centering then sketch.
        X = key_states - key_states.mean(dim=-2, keepdim=True)
        X = torch.matmul(X, Phi).to(torch.float32)  # (B,H,S,k)
        XT = X.transpose(-2, -1)  # (B,H,k,S)
        G = XT @ X  # (X^T X) / (B,H,k,k)
        # After sketching, we want to compute leverage scores given by
        # diag(X (X^T X)^{-1} X^T). But we don't want to form (X^T X)^{-1}
        # explicitly because it is slow and numerically unstable, so we
        # instead compute a Cholesky decomp G = (X^T X) = LL^T.
        L = LeverageScorePress.chol_with_jitter(0.5 * (G + G.transpose(-2, -1)), jitter=1e-2, max_tries=5)  # (B,H,k,k)

        # we use torch.cholesky_solve(XT, L) to find Y such that GY = X^T
        # given a cholesky factor L of G=LL^T (i.e we find Y = G^{-1}X^T)
        inv_Xt = torch.cholesky_solve(XT, L, upper=False)  # (X^TX)^{-1} X^T / (B,H,k,S)
        # we can now compute the leverage scores as: diag(X (X^T X)^{-1} X^T)
        # without materializing the full S x S matrix.
        scores = (X * inv_Xt.transpose(-2, -1)).sum(dim=-1).clamp_min(0)  # (B,H,S)
        return scores

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
        assert keys.shape[-2] == n_queries, "LeverageScorePress only supports prefill "
        # pre-RoPE keys from the hidden states for the current layer
        pre_rope_keys = get_prerope_key_states(module, hidden_states)  # (B,H_kv,S,d)
        scores = self.compute_leverage_scores(pre_rope_keys, self.sketch_dimension)  # (B,H_kv,S)
        z_scores = (scores - scores.mean()) / scores.std().clamp_min(1e-6)
        return z_scores
